/**
 * Teste de ponta a ponta, sem chave de LLM: o grafo de verdade, com um modelo
 * roteirizado, fala com o serviço Python no ar, o índice real e a API pública do npm.
 *
 * Fica fora de `pnpm test` porque depende de serviço e de rede:
 *   1. suba o serviço: `uv run --all-extras bula-knowledge serve` (com o índice ingerido)
 *   2. rode: `pnpm test:live`
 */
import { AIMessage, type BaseMessage } from "@langchain/core/messages";
import { describe, expect, it } from "vitest";
import { Agent } from "../../src/agent/agent.js";
import { MESSAGES } from "../../src/agent/guardrails.js";
import { KnowledgeClient } from "../../src/knowledge/client.js";
import { buildTools } from "../../src/tools/index.js";
import { call, ScriptedChatModel, type Step } from "../fakes.js";

const knowledge = new KnowledgeClient({
  baseUrl: process.env.KNOWLEDGE_URL ?? "http://127.0.0.1:8100",
  token: process.env.KNOWLEDGE_API_TOKEN,
  timeoutMs: 30_000,
});

function agentWith(steps: Step[], maxSteps = 4): Agent {
  return new Agent({
    model: new ScriptedChatModel(steps),
    tools: buildTools({ knowledge, timeoutMs: 15_000 }),
    systemPrompt: "live",
    maxSteps,
    maxTurns: 5,
  });
}

/** Faz o papel do modelo que leu o resultado: cita a primeira fonte numerada que recebeu. */
const citeFirstSource: Step = (messages: BaseMessage[]) => {
  const number = /\[(\d+)\] /.exec(messages.at(-1)?.text ?? "")?.[1];
  return new AIMessage(number ? `Resposta sustentada pela fonte [${number}].` : "Sem fonte.");
};

describe("ponta a ponta com o serviço de conhecimento no ar", () => {
  it("responde com citação que aponta para a documentação indexada", async () => {
    const agent = agentWith([
      call(["search_docs", { question: "How do I run code before every request in Fastify?" }]),
      citeFirstSource,
    ]);
    const result = await agent.ask("Como executo um código antes de cada requisição no Fastify?");

    expect(result.outcome).toBe("answered");
    expect(result.citations).toHaveLength(1);
    expect(result.citations[0]?.kind).toBe("docs");
    expect(result.citations[0]?.url).toMatch(
      /^https:\/\/github\.com\/fastify\/fastify\/blob\/[0-9a-f]{40}\/docs\//,
    );
  });

  it("diz que não encontrou quando a pergunta está fora do corpus", async () => {
    const agent = agentWith([
      call(["search_docs", { question: "receita de bolo de cenoura" }]),
      citeFirstSource,
    ]);
    const result = await agent.ask("Qual a receita de bolo de cenoura?");
    expect(result.outcome).toBe("not_found");
    expect(result.answer).toBe(MESSAGES.not_found);
  });

  it("consulta o npm de verdade para uma pergunta de versão", async () => {
    const agent = agentWith([call(["npm_package", { name: "fastify" }]), citeFirstSource]);
    const result = await agent.ask("Qual a versão mais recente do fastify?");
    expect(result.outcome).toBe("answered");
    expect(result.citations).toEqual([
      { n: 1, kind: "npm", title: "npm: fastify", url: "https://www.npmjs.com/package/fastify" },
    ]);
  });

  it("para no teto de passos quando o modelo insiste em consultar", async () => {
    const agent = agentWith([call(["search_docs", { question: "fastify hooks" }])], 3);
    const result = await agent.ask("pergunta que nunca fecha");
    expect(result.outcome).toBe("step_limit");
    expect(result.steps).toBe(3);
  });
});
