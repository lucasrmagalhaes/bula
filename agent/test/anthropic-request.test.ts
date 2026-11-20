/**
 * Fixa o que o agente manda para a API da Anthropic.
 *
 * Um servidor HTTP local faz o papel da API e guarda o pedido. O teste passa pelo
 * adaptador do LangChain e pelo SDK de verdade, sem chave e sem rede externa.
 */
import { createServer, type IncomingHttpHeaders, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { HumanMessage } from "@langchain/core/messages";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";
import { loadConfig } from "../src/config.js";
import { createChatModel, FALLBACK_BETA } from "../src/llm/models.js";

interface Captured {
  url: string;
  headers: IncomingHttpHeaders;
  body: Record<string, unknown>;
}

let server: Server;
let baseUrl: string;
let captured: Captured[];

const reply = {
  id: "msg_test",
  type: "message",
  role: "assistant",
  model: "claude-opus-5-5",
  content: [
    { type: "tool_use", id: "toolu_1", name: "search_docs", input: { question: "fastify hooks" } },
  ],
  stop_reason: "tool_use",
  stop_sequence: null,
  usage: { input_tokens: 120, output_tokens: 18 },
};

beforeEach(async () => {
  captured = [];
  server = createServer((request, response) => {
    let raw = "";
    request.on("data", (chunk: Buffer) => (raw += chunk.toString()));
    request.on("end", () => {
      captured.push({
        url: request.url ?? "",
        headers: request.headers,
        body: JSON.parse(raw) as Record<string, unknown>,
      });
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify(reply));
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

afterEach(async () => {
  vi.unstubAllEnvs();
  await new Promise((resolve) => server.close(resolve));
});

const tools = [
  {
    name: "search_docs",
    description: "Searches the documentation.",
    schema: z.object({ question: z.string() }),
  },
];

async function invoke(env: Record<string, string> = {}) {
  const config = loadConfig({
    LLM_PROVIDER: "anthropic",
    ANTHROPIC_API_KEY: "chave-de-teste",
    ANTHROPIC_API_URL: baseUrl,
    LLM_MAX_RETRIES: "0",
    ...env,
  });
  const model = await createChatModel(config.llm);
  const bound = model.bindTools?.(tools) ?? model;
  const response = await bound.invoke([new HumanMessage("Quando roda o onRequest?")]);
  const request = captured[0];
  if (!request) throw new Error("nenhum pedido chegou ao servidor de teste");
  return { response, request };
}

describe("pedido à API da Anthropic", () => {
  it("não manda amostragem nem tool_choice forçado, e usa o modelo padrão", async () => {
    // Um ANTHROPIC_BASE_URL solto no ambiente não pode desviar a chamada.
    vi.stubEnv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9/nao-usar");
    const { request, response } = await invoke();

    expect(request.url).toMatch(/^\/v1\/messages/);
    expect(request.body.model).toBe("claude-opus-5-5");
    expect(request.body.max_tokens).toBe(16_000);
    expect(Object.keys(request.body)).not.toEqual(
      expect.arrayContaining(["temperature", "top_p", "top_k"]),
    );
    const choice = request.body.tool_choice as { type?: string } | undefined;
    expect(choice === undefined || choice.type === "auto").toBe(true);
    const sent = request.body.tools as { name: string; input_schema: { properties: object } }[];
    expect(sent.map((tool) => tool.name)).toEqual(["search_docs"]);
    expect(sent[0]?.input_schema.properties).toHaveProperty("question");

    // A chamada de tool volta interpretada e com o uso de tokens.
    expect(response.tool_calls).toMatchObject([
      { name: "search_docs", args: { question: "fastify hooks" } },
    ]);
    expect(response.usage_metadata).toMatchObject({ input_tokens: 120, output_tokens: 18 });
  });

  it("liga o fallback de recusa do lado do servidor por padrão", async () => {
    const { request } = await invoke();
    expect(request.body.fallbacks).toBe("default");
    expect(String(request.headers["anthropic-beta"])).toContain(FALLBACK_BETA);
  });

  it("não manda o fallback quando desligado nem em modelo que não o aceita", async () => {
    const off = await invoke({ ANTHROPIC_FALLBACKS: "false" });
    expect(off.request.body).not.toHaveProperty("fallbacks");
    captured = [];
    const haiku = await invoke({ LLM_MODEL: "claude-haiku-5-5" });
    expect(haiku.request.body.model).toBe("claude-haiku-5-5");
    expect(haiku.request.body).not.toHaveProperty("fallbacks");
  });

  it("manda o esforço só quando configurado", async () => {
    const standard = await invoke();
    expect(standard.request.body).not.toHaveProperty("output_config");
    captured = [];
    const low = await invoke({ AGENT_EFFORT: "low" });
    expect(low.request.body.output_config).toMatchObject({ effort: "low" });
  });
});
