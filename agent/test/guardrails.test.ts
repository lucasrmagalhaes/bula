import { describe, expect, it } from "vitest";
import {
  DIRECT_ANSWER_MAX_CHARS,
  judge,
  MESSAGES,
  type Source,
} from "../src/agent/guardrails.js";

const source = (n: number): Source => ({
  n,
  kind: "docs",
  title: `Fonte ${n}`,
  url: `https://example.test/${n}`,
  key: String(n),
});

const base = { sources: [source(1), source(2)], turnSourceStart: 0, toolCalls: 1, toolErrors: 0 };

describe("guardrail de saída", () => {
  it("libera a resposta citada e devolve só as fontes usadas, em ordem", () => {
    const verdict = judge({ ...base, text: "O hook roda antes [2]. E depois valida [1][2]." });
    expect(verdict.outcome).toBe("answered");
    expect(verdict.citations.map((citation) => citation.n)).toEqual([1, 2]);
    expect(verdict.answer).toBe("O hook roda antes [2]. E depois valida [1][2].");
  });

  it("remove a citação que aponta para fonte inexistente", () => {
    const verdict = judge({ ...base, text: "Isto está na doc [1] e isto foi inventado [9]." });
    expect(verdict.outcome).toBe("answered");
    expect(verdict.answer).toBe("Isto está na doc [1] e isto foi inventado.");
  });

  it("retém a resposta que só cita fonte inexistente", () => {
    const verdict = judge({ ...base, text: "Segundo a documentação [7], funciona assim." });
    expect(verdict).toEqual({ outcome: "blocked", answer: MESSAGES.blocked, citations: [] });
  });

  it("retém a resposta sem citação quando havia fontes", () => {
    expect(judge({ ...base, text: "Funciona assim, pode confiar." }).outcome).toBe("blocked");
  });

  it("troca por 'não encontrei' quando houve consulta e nenhuma fonte voltou", () => {
    const verdict = judge({ ...base, sources: [], text: "Acho que é assim, de memória." });
    expect(verdict).toEqual({ outcome: "not_found", answer: MESSAGES.not_found, citations: [] });
  });

  it("avisa da falha quando a consulta falhou", () => {
    const verdict = judge({ ...base, sources: [], toolErrors: 1, text: "Desculpe." });
    expect(verdict.outcome).toBe("tool_error");
  });

  it("aceita citar fonte de um turno anterior mesmo sem fonte nova", () => {
    const verdict = judge({ ...base, turnSourceStart: 2, text: "Como visto antes [1]." });
    expect(verdict.outcome).toBe("answered");
  });

  it("sem consulta, só passa resposta curta", () => {
    const short = judge({ ...base, sources: [], toolCalls: 0, text: "Olá! Como posso ajudar?" });
    expect(short.outcome).toBe("direct");
    const long = judge({
      ...base,
      sources: [],
      toolCalls: 0,
      text: "x".repeat(DIRECT_ANSWER_MAX_CHARS + 1),
    });
    // A mensagem diz o motivo certo: não houve consulta, não "as fontes não sustentam".
    expect(long).toEqual({
      outcome: "blocked",
      answer: MESSAGES.blocked_no_lookup,
      citations: [],
    });
    expect(judge({ ...base, sources: [], toolCalls: 0, text: "  " }).outcome).toBe("blocked");
  });
});
