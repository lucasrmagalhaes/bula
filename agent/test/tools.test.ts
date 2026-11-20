import { describe, expect, it } from "vitest";
import { KnowledgeError } from "../src/knowledge/client.js";
import { searchDocsTool } from "../src/tools/docs.js";
import { githubReleaseTool } from "../src/tools/github.js";
import { npmPackageTool } from "../src/tools/npm.js";
import { ToolError } from "../src/tools/types.js";
import { contextOf, FakeKnowledge, fakeFetch, HOOKS_CHUNK, timeoutError } from "./fakes.js";

const context = { requestId: "req-1" };
const latest = { name: "fastify", version: "5.12.8", license: "MIT", description: "Fast web framework" };
const downloads = { downloads: 4_200_000, start: "2026-11-12", end: "2026-11-18" };

describe("npm_package", () => {
  it("devolve versão, licença e downloads como fonte citável", async () => {
    const fetch = fakeFetch({
      "registry.npmjs.org/fastify/latest": { body: latest },
      "api.npmjs.org/downloads/point/last-week/fastify": { body: downloads },
    });
    const result = await npmPackageTool({ timeoutMs: 500, fetch }).run({ name: " Fastify " }, context);
    expect(result.items).toHaveLength(1);
    expect(result.items[0]).toMatchObject({
      kind: "npm",
      title: "npm: fastify",
      url: "https://www.npmjs.com/package/fastify",
    });
    expect(result.items[0]?.body).toContain("Latest version: 5.12.8");
    expect(result.items[0]?.body).toContain("Downloads from 2026-11-12 to 2026-11-18: 4200000");
  });

  it("codifica o pacote com escopo na URL do registro", async () => {
    const fetch = fakeFetch({ "%2Fcore/latest": { body: { name: "@langchain/core", version: "1.4.2" } } });
    await npmPackageTool({ timeoutMs: 500, fetch }).run({ name: "@langchain/core" }, context);
    expect(fetch.urls[0]).toBe("https://registry.npmjs.org/@langchain%2Fcore/latest");
  });

  it("recusa nome inválido sem chamar a rede", async () => {
    const fetch = fakeFetch({});
    const result = await npmPackageTool({ timeoutMs: 500, fetch }).run(
      { name: "../../etc/passwd" },
      context,
    );
    expect(result.items).toEqual([]);
    expect(result.note).toContain("not a valid npm package name");
    expect(fetch.urls).toEqual([]);
  });

  it("trata 404 como pacote inexistente, não como falha", async () => {
    const result = await npmPackageTool({ timeoutMs: 500, fetch: fakeFetch({}) }).run(
      { name: "pacote-que-nao-existe" },
      context,
    );
    expect(result).toEqual({ items: [], note: 'No package named "pacote-que-nao-existe" on npm.' });
  });

  it("ainda responde a versão quando a API de downloads falha", async () => {
    const fetch = fakeFetch({
      "registry.npmjs.org": { body: latest },
      "api.npmjs.org": { status: 500 },
    });
    const result = await npmPackageTool({ timeoutMs: 500, fetch }).run({ name: "fastify" }, context);
    expect(result.items[0]?.body).toContain("Downloads: unavailable right now");
  });

  it.each([
    ["timeout", timeoutError(), "timeout"],
    ["rede", new TypeError("fetch failed"), "network"],
  ])("converte falha de %s em erro tipado", async (_label, failure, kind) => {
    const tool = npmPackageTool({ timeoutMs: 500, fetch: fakeFetch({ "registry.npmjs.org": failure }) });
    const error = await tool.run({ name: "fastify" }, context).catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(ToolError);
    expect((error as ToolError).kind).toBe(kind);
  });

  it.each([
    ["limite de requisições", { status: 429 }, "rate_limit"],
    ["erro do servidor", { status: 503 }, "upstream"],
    ["formato inesperado", { body: { nome: "x" } }, "shape"],
  ])("converte %s em erro tipado", async (_label, route, kind) => {
    const tool = npmPackageTool({ timeoutMs: 500, fetch: fakeFetch({ "registry.npmjs.org": route }) });
    await expect(tool.run({ name: "fastify" }, context)).rejects.toMatchObject({ kind });
  });
});

describe("github_latest_release", () => {
  const release = {
    tag_name: "v5.12.8",
    name: "v5.12.8",
    html_url: "https://github.com/fastify/fastify/releases/tag/v5.12.8",
    published_at: "2026-11-10T12:00:00Z",
    body: "x".repeat(4_000),
  };

  it("devolve a release com as notas truncadas e manda o token quando há", async () => {
    const fetch = fakeFetch({ "repos/fastify/fastify/releases/latest": { body: release } });
    const result = await githubReleaseTool({ timeoutMs: 500, fetch }, "token-de-teste").run(
      { repository: "fastify/fastify" },
      context,
    );
    expect(result.items[0]).toMatchObject({ kind: "github", url: release.html_url });
    expect(result.items[0]?.body).toContain("Release notes (truncated):");
    expect(result.items[0]?.body.length).toBeLessThan(1_700);
    expect(fetch.headers[0]?.authorization).toBe("Bearer token-de-teste");
  });

  it("recusa repositório fora do formato sem chamar a rede", async () => {
    const fetch = fakeFetch({});
    const result = await githubReleaseTool({ timeoutMs: 500, fetch }).run(
      { repository: "fastify/fastify/../../users" },
      context,
    );
    expect(result.note).toContain('not in the "owner/name" format');
    expect(fetch.urls).toEqual([]);
  });

  it("trata 404 como repositório sem release", async () => {
    const result = await githubReleaseTool({ timeoutMs: 500, fetch: fakeFetch({}) }).run(
      { repository: "fastify/fastify" },
      context,
    );
    expect(result.note).toBe("No published release found for fastify/fastify.");
  });
});

describe("search_docs", () => {
  it("mapeia os trechos aprovados e repassa o uso de tokens do workflow", async () => {
    const tool = searchDocsTool(new FakeKnowledge(() => contextOf([HOOKS_CHUNK])));
    const result = await tool.run({ question: "onRequest" }, context);
    expect(result.items).toEqual([
      { kind: "docs", title: "Hooks › onRequest", url: HOOKS_CHUNK.url, body: HOOKS_CHUNK.content },
    ]);
    expect(result.usage).toEqual({ inputTokens: 300, outputTokens: 40, llmCalls: 2 });
  });

  it("antepõe o título do documento quando a seção não começa por ele", async () => {
    const chunk = { ...HOOKS_CHUNK, title: "Getting Started", heading: "Install" };
    const tool = searchDocsTool(new FakeKnowledge(() => contextOf([chunk])));
    expect((await tool.run({ question: "x" }, context)).items[0]?.title).toBe(
      "Getting Started › Install",
    );
  });

  it("avisa o modelo quando nada foi encontrado", async () => {
    const tool = searchDocsTool(new FakeKnowledge(() => contextOf([])));
    const result = await tool.run({ question: "ingress" }, context);
    expect(result.items).toEqual([]);
    expect(result.note).toContain("No relevant passage");
  });

  it("converte falha do serviço em erro de tool sem detalhe interno", async () => {
    const tool = searchDocsTool(
      new FakeKnowledge(() => {
        throw new KnowledgeError("http", "serviço de conhecimento respondeu 500", 500);
      }),
    );
    await expect(tool.run({ question: "x" }, context)).rejects.toMatchObject({
      kind: "unavailable",
      message: "a busca na documentação está indisponível",
    });
  });
});
