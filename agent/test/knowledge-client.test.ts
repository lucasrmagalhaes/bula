import { describe, expect, it } from "vitest";
import { KnowledgeClient } from "../src/knowledge/client.js";
import { fakeFetch, timeoutError } from "./fakes.js";

// Resposta real de POST /v1/context do serviço Python, com um campo a mais de propósito.
const CONTEXT = {
  status: "ok",
  question: "How do I validate an email address with Zod?",
  queries: ["How do I validate an email address with Zod?"],
  attempts: 1,
  chunks: [
    {
      id: 911,
      source: "zod",
      title: "Defining schemas",
      heading: "String formats › Emails",
      url: "https://github.com/colinhacks/zod/blob/b500a2e/packages/docs/content/api.mdx#emails",
      content: "To validate email addresses: z.email();",
      score: 0.0325,
      vector_rank: 1,
      lexical_rank: 1,
      vector_distance: 0.281,
    },
  ],
  mode: "offline",
  degraded: false,
  usage: { llm_calls: 0, input_tokens: 0, output_tokens: 0 },
  timings_ms: { rewrite: 0, retrieve: 48, grade: 0, total: 57.2 },
  trace_id: null,
};

const client = (fetch: typeof globalThis.fetch, token?: string) =>
  new KnowledgeClient({ baseUrl: "http://knowledge.test", token, timeoutMs: 500, fetch });

describe("cliente do serviço de conhecimento", () => {
  it("valida a resposta contra o contrato e descarta campos que não usa", async () => {
    const fetch = fakeFetch({ "/v1/context": { body: CONTEXT } });
    const result = await client(fetch, "segredo").context("email com zod", { requestId: "req-9" });

    expect(result.status).toBe("ok");
    expect(result.chunks[0]).toEqual({
      id: 911,
      source: "zod",
      title: "Defining schemas",
      heading: "String formats › Emails",
      url: CONTEXT.chunks[0]?.url,
      content: "To validate email addresses: z.email();",
      score: 0.0325,
    });
    expect(fetch.urls).toEqual(["http://knowledge.test/v1/context"]);
    expect(fetch.headers[0]).toMatchObject({
      authorization: "Bearer segredo",
      "x-request-id": "req-9",
      "content-type": "application/json",
    });
  });

  it("acusa quebra de contrato em vez de devolver dado incompleto", async () => {
    const broken = { ...CONTEXT, chunks: [{ id: 1, title: "sem url" }] };
    const fetch = fakeFetch({ "/v1/context": { body: broken } });
    await expect(client(fetch).context("x")).rejects.toMatchObject({
      name: "KnowledgeError",
      kind: "contract",
    });
  });

  it.each([
    ["401 do serviço", { status: 401 }, { kind: "http", status: 401 }],
    ["serviço fora do ar", new TypeError("fetch failed"), { kind: "unavailable" }],
    ["tempo esgotado", timeoutError(), { kind: "timeout" }],
  ])("classifica %s", async (_label, route, expected) => {
    const fetch = fakeFetch({ "/v1/context": route });
    await expect(client(fetch).context("x")).rejects.toMatchObject(expected);
  });

  it("lê a lista do que está indexado", async () => {
    const fetch = fakeFetch({
      "/v1/sources": {
        body: {
          embedding_model: "m",
          sources: [
            {
              id: "zod",
              title: "Zod",
              repo: "colinhacks/zod",
              ref: "v4.6.7",
              commit_sha: "b500a2e",
              license: "MIT",
              documents: 17,
              chunks: 357,
              ingested_at: "2026-11-20T01:09:20+00:00",
            },
          ],
        },
      },
    });
    const { sources } = await client(fetch).sources();
    expect(sources[0]).toMatchObject({ title: "Zod", ref: "v4.6.7", license: "MIT" });
  });
});
