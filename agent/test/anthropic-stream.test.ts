/**
 * O caminho que o servidor usa de verdade: `Agent.stream` com o adaptador da Anthropic
 * em streaming. Um servidor local responde em SSE, no formato da API, com um bloco de
 * raciocínio assinado e uma chamada de tool.
 *
 * O que isto fixa: os argumentos da tool chegam inteiros a partir dos pedaços de JSON,
 * o uso de tokens é contado, e o bloco de raciocínio volta para a API sem alteração nos
 * pedidos seguintes. Os modelos atuais recusam histórico em que esse bloco foi mexido.
 */
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { Agent, type AgentEvent } from "../src/agent/agent.js";
import { loadConfig } from "../src/config.js";
import { createChatModel } from "../src/llm/models.js";
import { buildTools } from "../src/tools/index.js";
import { contextOf, FakeKnowledge, fakeFetch, HOOKS_CHUNK } from "./fakes.js";

interface Block {
  type: string;
  [key: string]: unknown;
}

interface Body {
  stream?: boolean;
  messages: { role: string; content: string | Block[] }[];
}

let server: Server;
let baseUrl: string;
let bodies: Body[];

const THINKING = { type: "thinking", thinking: "Preciso consultar a documentação.", signature: "sig-abc123" };

function sse(events: Record<string, unknown>[]): string {
  return events.map((event) => `event: ${String(event.type)}\ndata: ${JSON.stringify(event)}\n\n`).join("");
}

function start(outputTokens = 1): Record<string, unknown> {
  return {
    type: "message_start",
    message: {
      id: "msg_test",
      type: "message",
      role: "assistant",
      model: "claude-opus-5-5",
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: 120, output_tokens: outputTokens },
    },
  };
}

function end(stopReason: string, outputTokens: number): Record<string, unknown>[] {
  return [
    {
      type: "message_delta",
      delta: { stop_reason: stopReason, stop_sequence: null },
      usage: { output_tokens: outputTokens },
    },
    { type: "message_stop" },
  ];
}

/** Primeira resposta: raciocínio assinado e uma chamada de tool com o JSON em dois pedaços. */
const toolTurn = sse([
  start(),
  { type: "content_block_start", index: 0, content_block: { type: "thinking", thinking: "", signature: "" } },
  { type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: THINKING.thinking } },
  { type: "content_block_delta", index: 0, delta: { type: "signature_delta", signature: THINKING.signature } },
  { type: "content_block_stop", index: 0 },
  {
    type: "content_block_start",
    index: 1,
    content_block: { type: "tool_use", id: "toolu_1", name: "search_docs", input: {} },
  },
  { type: "content_block_delta", index: 1, delta: { type: "input_json_delta", partial_json: '{"question": "fast' } },
  { type: "content_block_delta", index: 1, delta: { type: "input_json_delta", partial_json: 'ify hooks"}' } },
  { type: "content_block_stop", index: 1 },
  ...end("tool_use", 30),
]);

function textTurn(...pieces: string[]): string {
  return sse([
    start(),
    { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
    ...pieces.map((text) => ({
      type: "content_block_delta",
      index: 0,
      delta: { type: "text_delta", text },
    })),
    { type: "content_block_stop", index: 0 },
    ...end("end_turn", 12),
  ]);
}

beforeEach(async () => {
  bodies = [];
  const replies = [toolTurn, textTurn("Roda antes ", "do parse [1]."), textTurn("Sim, antes do parse [1].")];
  server = createServer((request, response) => {
    let raw = "";
    request.on("data", (chunk: Buffer) => (raw += chunk.toString()));
    request.on("end", () => {
      bodies.push(JSON.parse(raw) as Body);
      response.writeHead(200, { "content-type": "text/event-stream" });
      // Cada resposta tem um id próprio, como na API: o LangGraph junta mensagens pelo id.
      const reply = replies[bodies.length - 1] ?? textTurn("fim");
      response.end(reply.replaceAll("msg_test", `msg_test_${bodies.length}`));
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

afterEach(async () => {
  await new Promise((resolve) => server.close(resolve));
});

function blocksOf(body: Body | undefined, role: string): Block[] {
  return (body?.messages ?? [])
    .filter((message) => message.role === role)
    .flatMap((message) => (typeof message.content === "string" ? [] : message.content));
}

describe("streaming com a Anthropic, pelo caminho do servidor", () => {
  it("monta a chamada de tool, conta os tokens e devolve o raciocínio intacto", async () => {
    const config = loadConfig({
      LLM_PROVIDER: "anthropic",
      ANTHROPIC_API_KEY: "chave-de-teste",
      ANTHROPIC_API_URL: baseUrl,
      LLM_MAX_RETRIES: "0",
    });
    const knowledge = new FakeKnowledge(() => contextOf([HOOKS_CHUNK]));
    const agent = new Agent({
      model: await createChatModel(config.llm),
      tools: buildTools({ knowledge, timeoutMs: 500, fetch: fakeFetch({}) }),
      systemPrompt: "prompt de teste",
      maxSteps: 4,
      maxTurns: 5,
    });
    const threadId = "0b9d6c3e-6f0a-4c59-9d5e-0d7b1f0a00aa";

    const events: AgentEvent[] = [];
    for await (const event of agent.stream("Quando roda o onRequest?", { threadId })) {
      events.push(event);
    }
    const done = events.at(-1);
    if (done?.type !== "done") throw new Error("o turno não terminou");

    // O servidor usa streaming, e os argumentos da tool foram montados dos pedaços.
    expect(bodies[0]?.stream).toBe(true);
    expect(knowledge.questions).toEqual(["fastify hooks"]);
    expect(done.result).toMatchObject({ outcome: "answered", answer: "Roda antes do parse [1]." });
    const tokens = events.flatMap((event) => (event.type === "token" ? [event.text] : []));
    expect(tokens).toEqual(["Roda antes ", "do parse [1]."]);

    // Tokens das duas chamadas ao modelo, mais os do workflow de RAG que vêm pela tool.
    expect(done.result.usage.llmCalls).toBe(4);
    expect(done.result.usage.inputTokens).toBe(120 + 120 + 300);
    expect(done.result.usage.outputTokens).toBeGreaterThan(40);

    // Segundo pedido do mesmo turno: o raciocínio volta igual, junto da chamada e do resultado.
    const second = blocksOf(bodies[1], "assistant");
    expect(second[0]).toEqual(THINKING);
    expect(second[1]).toMatchObject({
      type: "tool_use",
      id: "toolu_1",
      name: "search_docs",
      input: { question: "fastify hooks" },
    });
    expect(blocksOf(bodies[1], "user").at(-1)).toMatchObject({
      type: "tool_result",
      tool_use_id: "toolu_1",
    });

    // Turno seguinte da conversa: o histórico só cresceu, e o bloco continua lá, intacto.
    const next = await agent.ask("Tem certeza?", { threadId });
    expect(next.outcome).toBe("answered");
    const third = bodies[2];
    expect(third?.messages.map((message) => message.role)).toEqual([
      "user",
      "assistant",
      "user",
      "assistant",
      "user",
    ]);
    expect(blocksOf(third, "assistant")[0]).toEqual(THINKING);
    expect(third?.messages.slice(0, 3)).toEqual(bodies[1]?.messages);
  });
});
