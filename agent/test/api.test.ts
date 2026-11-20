import type { FastifyInstance } from "fastify";
import { afterEach, describe, expect, it } from "vitest";
import { Agent } from "../src/agent/agent.js";
import { loadConfig, type Config } from "../src/config.js";
import { buildApp } from "../src/server/app.js";
import { buildTools } from "../src/tools/index.js";
import {
  call,
  contextOf,
  FakeKnowledge,
  fakeFetch,
  HOOKS_CHUNK,
  say,
  ScriptedChatModel,
  type Step,
} from "./fakes.js";

let app: FastifyInstance | undefined;

afterEach(async () => {
  await app?.close();
  app = undefined;
});

async function start(steps: Step[], env: Record<string, string> = {}): Promise<FastifyInstance> {
  const config: Config = loadConfig({ LOG_LEVEL: "silent", ...env });
  const agent = new Agent({
    model: new ScriptedChatModel(steps),
    tools: buildTools({
      knowledge: new FakeKnowledge(() => contextOf([HOOKS_CHUNK])),
      timeoutMs: 500,
      fetch: fakeFetch({}),
    }),
    systemPrompt: "p",
    maxSteps: config.agent.maxSteps,
    maxTurns: config.agent.maxTurns,
  });
  app = await buildApp({ agent, config, indexed: ["Fastify v5.12.8"] });
  return app;
}

const answered: Step[] = [call(["search_docs", { question: "hooks" }]), say("Antes do parse [1].")];

describe("API HTTP", () => {
  it("GET /health informa o modo e o índice", async () => {
    const response = await (await start([say("ok")])).inject({ url: "/health" });
    expect(response.json()).toEqual({
      status: "ok",
      llm: { provider: "offline", model: null },
      indexed: ["Fastify v5.12.8"],
    });
  });

  it("POST /v1/chat responde com fontes e devolve o id da conversa", async () => {
    const response = await (await start(answered)).inject({
      method: "POST",
      url: "/v1/chat",
      headers: { "x-request-id": "req-42" },
      payload: { message: "Quando roda o onRequest?" },
    });
    expect(response.statusCode).toBe(200);
    expect(response.headers["x-request-id"]).toBe("req-42");
    const body = response.json<{ threadId: string; requestId: string }>();
    expect(body).toMatchObject({
      outcome: "answered",
      answer: "Antes do parse [1].",
      requestId: "req-42",
      citations: [{ n: 1, kind: "docs", title: "Hooks › onRequest" }],
    });
    expect(body.threadId).toMatch(/^[0-9a-f-]{36}$/);
  });

  it.each([
    ["mensagem vazia", { message: "   " }],
    ["mensagem longa demais", { message: "x".repeat(2_001) }],
    ["threadId que não é uuid", { message: "oi", threadId: "../../etc" }],
    ["corpo sem mensagem", {}],
  ])("recusa %s com 400", async (_label, payload) => {
    const response = await (await start([say("ok")])).inject({
      method: "POST",
      url: "/v1/chat",
      payload,
    });
    expect(response.statusCode).toBe(400);
    expect(response.json()).toMatchObject({ error: "bad_request" });
  });

  it("devolve 409 quando a conversa chega ao limite de turnos", async () => {
    const server = await start([say("ok")], { AGENT_MAX_TURNS: "1" });
    const first = await server.inject({ method: "POST", url: "/v1/chat", payload: { message: "um" } });
    const { threadId } = first.json<{ threadId: string }>();
    const second = await server.inject({
      method: "POST",
      url: "/v1/chat",
      payload: { message: "dois", threadId },
    });
    expect(second.statusCode).toBe(409);
    expect(second.json()).toMatchObject({ error: "thread_full" });
  });

  it("limita a taxa de pedidos por cliente", async () => {
    const server = await start([say("ok")], { RATE_LIMIT_PER_MINUTE: "2" });
    const send = () => server.inject({ method: "POST", url: "/v1/chat", payload: { message: "oi" } });
    expect((await send()).statusCode).toBe(200);
    expect((await send()).statusCode).toBe(200);
    const blocked = await send();
    expect(blocked.statusCode).toBe(429);
    expect(blocked.json()).toMatchObject({ error: "rate_limited" });
  });

  it("POST /v1/chat/stream transmite passos, tokens e o resultado em SSE", async () => {
    const response = await (await start(answered)).inject({
      method: "POST",
      url: "/v1/chat/stream",
      payload: { message: "Quando roda o onRequest?" },
    });
    expect(response.headers["content-type"]).toBe("text/event-stream; charset=utf-8");
    const events = response.body
      .trim()
      .split("\n\n")
      .map((block) => {
        const [eventLine = "", dataLine = ""] = block.split("\n");
        return {
          event: eventLine.replace("event: ", ""),
          data: JSON.parse(dataLine.replace("data: ", "")) as Record<string, unknown>,
        };
      });
    expect(events[0]).toEqual({ event: "step", data: { node: "agent", tools: ["search_docs"] } });
    const tokens = events.filter((item) => item.event === "token").map((item) => item.data.text);
    expect(tokens.join("")).toBe("Antes do parse [1].");
    expect(events.at(-1)).toMatchObject({ event: "done", data: { outcome: "answered" } });
  });

  it("transmite por um socket de verdade, sem cortar ao terminar de ler o pedido", async () => {
    const server = await start(answered);
    const address = await server.listen({ host: "127.0.0.1", port: 0 });
    const response = await fetch(`${address}/v1/chat/stream`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message: "Quando roda o onRequest?" }),
    });
    const body = await response.text();
    expect(body).toContain("event: token");
    expect(body).toContain('event: done\ndata: {"threadId"');
    expect(body).toContain('"outcome":"answered"');
  });

  it("GET / serve a página de chat com política de conteúdo restrita", async () => {
    const response = await (await start([say("ok")])).inject({ url: "/" });
    expect(response.statusCode).toBe(200);
    expect(response.headers["content-type"]).toBe("text/html; charset=utf-8");
    expect(response.headers["content-security-policy"]).toContain("default-src 'none'");
    expect(response.body).toContain("<title>Bula</title>");
  });
});
