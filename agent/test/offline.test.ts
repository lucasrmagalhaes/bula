import { HumanMessage, ToolMessage } from "@langchain/core/messages";
import { describe, expect, it } from "vitest";
import { respond } from "../src/llm/offline.js";

const ask = (text: string) => respond([new HumanMessage(text)]);

describe("modo offline", () => {
  it("manda pergunta de documentação para a busca", () => {
    const question = "Como executo um código antes de cada requisição no Fastify?";
    expect(ask(question).tool_calls).toMatchObject([
      { name: "search_docs", args: { question } },
    ]);
  });

  it("manda pergunta de versão para o npm", () => {
    expect(ask("Qual a versão mais recente do pacote fastify?").tool_calls).toMatchObject([
      { name: "npm_package", args: { name: "fastify" } },
    ]);
    expect(ask("latest version of @langchain/core").tool_calls).toMatchObject([
      { name: "npm_package", args: { name: "@langchain/core" } },
    ]);
  });

  it("manda pergunta de release com repositório para o GitHub", () => {
    expect(ask("O que mudou na última release de fastify/fastify?").tool_calls).toMatchObject([
      { name: "github_latest_release", args: { repository: "fastify/fastify" } },
    ]);
  });

  it("responde saudação sem tool", () => {
    const message = ask("Oi!");
    expect(message.tool_calls).toEqual([]);
    expect(message.text).toContain("documentação indexada");
  });

  it("devolve os trechos encontrados com a citação de cada um", () => {
    const result = new ToolMessage({
      tool_call_id: "offline-search_docs",
      content:
        '<result tool="search_docs">\n[3] Hooks › onRequest\nURL: https://example.test/a\nRuns before parsing.\n\nSecond paragraph.\n\n' +
        "[4] Hooks › preHandler\nURL: https://example.test/b\nRuns before the handler.\n</result>",
    });
    const message = respond([new HumanMessage("hooks?"), ask("hooks?"), result]);
    expect(message.text).toContain("Modo offline, sem modelo de linguagem");
    // O trecho inteiro entra no resumo, não só a primeira linha.
    expect(message.text).toContain("- Hooks › onRequest [3]: Runs before parsing. Second paragraph.");
    expect(message.text).toContain("- Hooks › preHandler [4]: Runs before the handler.");
  });

  it("diz que não achou quando a tool voltou vazia", () => {
    const empty = new ToolMessage({ tool_call_id: "x", content: "No relevant passage was found." });
    expect(respond([new HumanMessage("x"), ask("x"), empty]).text).toBe(
      "Não encontrei nada sobre isso nas fontes consultadas.",
    );
  });
});
