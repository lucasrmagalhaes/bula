import { describe, expect, it } from "vitest";
import { ConfigError, loadConfig } from "../src/config.js";

describe("configuração", () => {
  it("sem nada definido, sobe em modo offline com os padrões", () => {
    const config = loadConfig({});
    expect(config.llm).toMatchObject({ provider: "offline", model: "", maxRetries: 2 });
    expect(config.knowledge.url).toBe("http://127.0.0.1:8100");
    expect(config.server.port).toBe(8200);
    expect(config.agent).toEqual({ maxSteps: 6, maxTurns: 20, maxMessageChars: 2_000 });
  });

  it("trata valor em branco do .env como ausente", () => {
    const config = loadConfig({ LLM_PROVIDER: "", AGENT_PORT: " ", KNOWLEDGE_API_TOKEN: "" });
    expect(config.llm.provider).toBe("offline");
    expect(config.server.port).toBe(8200);
    expect(config.knowledge.token).toBeUndefined();
  });

  it("usa o modelo padrão só para a Anthropic", () => {
    const config = loadConfig({ LLM_PROVIDER: "anthropic", ANTHROPIC_API_KEY: "k" });
    expect(config.llm.model).toBe("claude-opus-5-5");
    expect(() => loadConfig({ LLM_PROVIDER: "openai", OPENAI_API_KEY: "k" })).toThrow(
      "LLM_MODEL é obrigatório com LLM_PROVIDER=openai",
    );
  });

  it("falha na subida quando falta a chave do provedor escolhido", () => {
    expect(() => loadConfig({ LLM_PROVIDER: "anthropic" })).toThrow(
      "ANTHROPIC_API_KEY é obrigatória",
    );
    expect(() => loadConfig({ LLM_PROVIDER: "google", LLM_MODEL: "m" })).toThrow(ConfigError);
  });

  it("recusa valores fora do intervalo e diz qual variável", () => {
    expect(() => loadConfig({ AGENT_MAX_STEPS: "0" })).toThrow(/AGENT_MAX_STEPS/);
    expect(() => loadConfig({ KNOWLEDGE_URL: "nao-e-url" })).toThrow(/KNOWLEDGE_URL/);
    expect(() => loadConfig({ LLM_PROVIDER: "llama" })).toThrow(/LLM_PROVIDER/);
  });

  it("só calcula custo quando os dois preços foram informados", () => {
    expect(loadConfig({ LLM_PRICE_INPUT_PER_MTOK: "4" }).llm.price).toBeUndefined();
    expect(
      loadConfig({ LLM_PRICE_INPUT_PER_MTOK: "4", LLM_PRICE_OUTPUT_PER_MTOK: "20" }).llm.price,
    ).toEqual({ inputPerMTok: 4, outputPerMTok: 20 });
  });

  it("remove a barra final da URL do serviço de conhecimento", () => {
    expect(loadConfig({ KNOWLEDGE_URL: "http://knowledge:8100/" }).knowledge.url).toBe(
      "http://knowledge:8100",
    );
  });
});
