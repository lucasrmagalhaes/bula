import { ToolMessage } from "@langchain/core/messages";
import { describe, expect, it } from "vitest";
import { Agent, ThreadFullError, type AgentEvent } from "../src/agent/agent.js";
import { MESSAGES } from "../src/agent/guardrails.js";
import { KnowledgeError } from "../src/knowledge/client.js";
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

const NPM_ROUTES = {
  "registry.npmjs.org/fastify/latest": {
    body: { name: "fastify", version: "5.12.8", license: "MIT" },
  },
  "api.npmjs.org/downloads": {
    body: { downloads: 123_456, start: "2026-11-12", end: "2026-11-18" },
  },
};

function setup(
  steps: Step[],
  options: {
    knowledge?: FakeKnowledge;
    maxSteps?: number;
    maxTurns?: number;
    price?: { inputPerMTok: number; outputPerMTok: number };
  } = {},
) {
  const model = new ScriptedChatModel(steps);
  const knowledge = options.knowledge ?? new FakeKnowledge(() => contextOf([HOOKS_CHUNK]));
  const failures: string[] = [];
  const agent = new Agent({
    model,
    tools: buildTools({ knowledge, timeoutMs: 1_000, fetch: fakeFetch(NPM_ROUTES) }),
    systemPrompt: "prompt de teste",
    maxSteps: options.maxSteps ?? 4,
    maxTurns: options.maxTurns ?? 10,
    price: options.price,
    onToolFailure: (tool) => failures.push(tool),
  });
  return { agent, model, knowledge, failures };
}

describe("agente", () => {
  it("busca na documentação e responde citando a fonte", async () => {
    const { agent, model, knowledge } = setup(
      [call(["search_docs", { question: "fastify onRequest hook" }]), say("Roda antes do parse [1].")],
      { price: { inputPerMTok: 4, outputPerMTok: 20 } },
    );

    const result = await agent.ask("Quando roda o onRequest?");

    expect(result.outcome).toBe("answered");
    expect(result.answer).toBe("Roda antes do parse [1].");
    expect(result.citations).toEqual([
      { n: 1, kind: "docs", title: "Hooks › onRequest", url: HOOKS_CHUNK.url },
    ]);
    expect(knowledge.questions).toEqual(["fastify onRequest hook"]);
    expect(result.steps).toBe(2);
    // Soma as duas chamadas do agente e as duas do workflow de RAG, que vêm pela tool.
    expect(result.usage).toMatchObject({ inputTokens: 450, outputTokens: 70, llmCalls: 4 });
    expect(result.usage.costUsd).toBeCloseTo((450 * 4 + 70 * 20) / 1e6, 9);

    // O modelo recebeu o prompt de sistema e o resultado numerado, marcado como dado.
    const secondCall = model.calls[1] ?? [];
    expect(secondCall[0]?.text).toBe("prompt de teste");
    const toolResult = secondCall.at(-1);
    expect(toolResult).toBeInstanceOf(ToolMessage);
    expect(toolResult?.text).toContain('<result tool="search_docs">\n[1] Hooks › onRequest\nURL: ');
    expect(model.boundTools.map((tool) => (tool as { name: string }).name)).toEqual([
      "search_docs",
      "npm_package",
      "github_latest_release",
    ]);
  });

  it("devolve a mensagem fixa de 'não encontrei' quando a busca não acha nada", async () => {
    const { agent } = setup(
      [call(["search_docs", { question: "x" }]), say("De memória, acho que é com um middleware.")],
      { knowledge: new FakeKnowledge(() => contextOf([])) },
    );
    const result = await agent.ask("Como configuro o ingress do Kubernetes?");
    expect(result.outcome).toBe("not_found");
    expect(result.answer).toBe(MESSAGES.not_found);
    expect(result.citations).toEqual([]);
  });

  it("retém a resposta que ignora as fontes encontradas", async () => {
    const { agent } = setup([call(["search_docs", { question: "x" }]), say("É assim, confie.")]);
    const result = await agent.ask("Quando roda o onRequest?");
    expect(result.outcome).toBe("blocked");
    expect(result.answer).toBe(MESSAGES.blocked);
  });

  it("para no teto de passos e deixa o histórico válido para o próximo turno", async () => {
    const loop = call(["search_docs", { question: "de novo" }]);
    const { agent, model } = setup([loop, loop, loop, say("Agora sim [1].")], { maxSteps: 3 });

    const first = await agent.ask("pergunta sem fim", { threadId: THREAD });
    expect(first.outcome).toBe("step_limit");
    expect(first.answer).toBe(MESSAGES.step_limit);
    expect(first.steps).toBe(3);
    expect(model.calls).toHaveLength(3);

    // A chamada de tool que ficou pendente ganhou um resultado; o turno seguinte funciona.
    const second = await agent.ask("e agora?", { threadId: THREAD });
    expect(second.outcome).toBe("answered");
    const history = model.calls[3] ?? [];
    const pending = history.at(-2);
    expect(pending).toBeInstanceOf(ToolMessage);
    expect(pending?.text).toContain("step limit");
  });

  it("avisa da falha quando o serviço de conhecimento está fora", async () => {
    const down = new FakeKnowledge(() => {
      throw new KnowledgeError("unavailable", "fora do ar");
    });
    const { agent, model, failures } = setup(
      [call(["search_docs", { question: "x" }]), say("Não consegui consultar.")],
      { knowledge: down },
    );
    const result = await agent.ask("Quando roda o onRequest?");
    expect(result.outcome).toBe("tool_error");
    expect(result.answer).toBe(MESSAGES.tool_error);
    expect(failures).toEqual(["search_docs"]);
    // O modelo vê a falha como resultado da tool, sem detalhe interno.
    expect(model.calls[1]?.at(-1)?.text).toBe("Tool failed: a busca na documentação está indisponível");
  });

  it("trata tool desconhecida e argumento inválido como erro de tool, sem quebrar", async () => {
    const { agent, model } = setup([
      call(["apagar_tudo", {}], ["npm_package", { nome: "fastify" }]),
      say("Não deu."),
    ]);
    const result = await agent.ask("teste");
    expect(result.outcome).toBe("tool_error");
    const results = (model.calls[1] ?? []).slice(-2).map((message) => message.text);
    expect(results[0]).toBe('Tool failed: unknown tool "apagar_tudo"');
    expect(results[1]).toMatch(/^Tool failed: invalid arguments/);
  });

  it("responde saudação sem consultar nada", async () => {
    const { agent, knowledge } = setup([say("Olá! Pergunte sobre a documentação.")]);
    const result = await agent.ask("oi");
    expect(result.outcome).toBe("direct");
    expect(knowledge.questions).toEqual([]);
  });

  it("numera em ordem as fontes de tools chamadas em paralelo", async () => {
    const { agent } = setup([
      call(["search_docs", { question: "hooks" }], ["npm_package", { name: "fastify" }]),
      say("O hook existe [1] e a versão atual é a 5.12.8 [2]."),
    ]);
    const result = await agent.ask("Tem hook onRequest? E qual a versão?");
    expect(result.outcome).toBe("answered");
    expect(result.citations.map(({ n, kind }) => [n, kind])).toEqual([
      [1, "docs"],
      [2, "npm"],
    ]);
    expect(result.citations[1]?.url).toBe("https://www.npmjs.com/package/fastify");
  });

  it("mantém a conversa: o segundo turno vê o primeiro e não renumera a mesma fonte", async () => {
    const { agent, model } = setup([
      call(["search_docs", { question: "hooks" }]),
      say("Roda antes do parse [1]."),
      call(["search_docs", { question: "hooks de novo" }]),
      say("Como dito, antes do parse [1]."),
    ]);
    await agent.ask("Quando roda o onRequest?", { threadId: THREAD_2 });
    const second = await agent.ask("Tem certeza?", { threadId: THREAD_2 });

    expect(second.outcome).toBe("answered");
    expect(second.citations.map((citation) => citation.n)).toEqual([1]);
    const lastCall = model.calls.at(-1) ?? [];
    expect(lastCall.map((message) => message.type)).toEqual([
      "system",
      "human",
      "ai",
      "tool",
      "ai",
      "human",
      "ai",
      "tool",
    ]);
    // O mesmo trecho volta com o número que já tinha.
    expect(lastCall.at(-1)?.text).toContain("[1] Hooks › onRequest");
  });

  it("recusa novo turno quando a conversa chega ao limite", async () => {
    const { agent } = setup([say("ok")], { maxTurns: 2 });
    await agent.ask("um", { threadId: THREAD_3 });
    await agent.ask("dois", { threadId: THREAD_3 });
    await expect(agent.ask("três", { threadId: THREAD_3 })).rejects.toBeInstanceOf(ThreadFullError);
  });

  it("transmite passos, tokens e o resultado final, nesta ordem", async () => {
    const { agent } = setup([
      call(["search_docs", { question: "hooks" }]),
      say("Roda antes do parse [1]."),
    ]);
    const events: AgentEvent[] = [];
    for await (const event of agent.stream("Quando roda o onRequest?")) events.push(event);

    const kinds = events.map((event) => (event.type === "step" ? `step:${event.node}` : event.type));
    expect(kinds.at(-1)).toBe("done");
    expect(kinds.filter((kind) => kind.startsWith("step"))).toEqual([
      "step:agent",
      "step:tools",
      "step:agent",
      "step:finish",
    ]);
    expect(events.find((event) => event.type === "step")).toMatchObject({ tools: ["search_docs"] });
    const streamed = events.flatMap((event) => (event.type === "token" ? [event.text] : []));
    expect(streamed.length).toBeGreaterThan(1);
    expect(streamed.join("")).toBe("Roda antes do parse [1].");
  });
});

const THREAD = "0b9d6c3e-6f0a-4c59-9d5e-0d7b1f0a0001";
const THREAD_2 = "0b9d6c3e-6f0a-4c59-9d5e-0d7b1f0a0002";
const THREAD_3 = "0b9d6c3e-6f0a-4c59-9d5e-0d7b1f0a0003";

describe("interrupção", () => {
  it("para o grafo quando o pedido é abortado, sem chamar o modelo de novo", async () => {
    const slow = new FakeKnowledge(async () => {
      await new Promise((resolve) => setTimeout(resolve, 150));
      return contextOf([HOOKS_CHUNK]);
    });
    const { agent, model } = setup(
      [call(["search_docs", { question: "hooks" }]), say("Roda antes do parse [1].")],
      { knowledge: slow },
    );
    const abort = new AbortController();

    const run = (async () => {
      for await (const event of agent.stream("Quando roda o onRequest?", { signal: abort.signal })) {
        // O cliente desiste assim que o agente pede a primeira tool.
        if (event.type === "step" && event.node === "agent") abort.abort();
      }
    })();

    await expect(run).rejects.toThrow();
    expect(model.calls).toHaveLength(1);
  });
});
