import { describe, expect, it } from "vitest";
import { loadIndexed } from "../src/bootstrap.js";
import {
  KnowledgeError,
  type KnowledgeApi,
  type KnowledgeErrorKind,
  type SourcesResponse,
} from "../src/knowledge/client.js";

const SOURCES: SourcesResponse = {
  embedding_model: "m",
  sources: [
    { id: "zod", title: "Zod", repo: "r", ref: "v4.6.7", license: "MIT", documents: 1, chunks: 1 },
  ],
};

/** Falha `failures` vezes com o tipo dado e depois responde. */
function flaky(failures: number, kind: KnowledgeErrorKind): KnowledgeApi & { calls: number } {
  const api = {
    calls: 0,
    context: () => Promise.reject(new Error("não usado")),
    sources: () => {
      api.calls += 1;
      return api.calls <= failures
        ? Promise.reject(new KnowledgeError(kind, "falha"))
        : Promise.resolve(SOURCES);
    },
  };
  return api;
}

describe("subida do agente", () => {
  it("espera o serviço de conhecimento quando os dois sobem juntos", async () => {
    const knowledge = flaky(2, "unavailable");
    const indexed = await loadIndexed(knowledge, { waitMs: 2_000, intervalMs: 5 });
    expect(indexed).toEqual(["Zod v4.6.7"]);
    expect(knowledge.calls).toBe(3);
  });

  it("desiste no prazo e segue sem a lista", async () => {
    const knowledge = flaky(Infinity, "unavailable");
    const gaveUp: string[] = [];
    const indexed = await loadIndexed(knowledge, {
      waitMs: 30,
      intervalMs: 5,
      onGiveUp: (kind) => gaveUp.push(kind),
    });
    expect(indexed).toEqual([]);
    expect(gaveUp).toEqual(["unavailable"]);
    expect(knowledge.calls).toBeGreaterThan(1);
  });

  it("não insiste em erro que esperar não resolve", async () => {
    const knowledge = flaky(Infinity, "http");
    expect(await loadIndexed(knowledge, { waitMs: 2_000, intervalMs: 5 })).toEqual([]);
    expect(knowledge.calls).toBe(1);
  });
});
