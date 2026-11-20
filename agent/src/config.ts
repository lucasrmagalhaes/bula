import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { z } from "zod";

/** `CHAVE=` no .env chega como string vazia; trata como ausente. */
const blankToUndefined = (value: unknown): unknown =>
  typeof value === "string" && value.trim() === "" ? undefined : value;

const text = z.preprocess(blankToUndefined, z.string().optional());
const int = (fallback: number, min: number, max: number) =>
  z.preprocess(blankToUndefined, z.coerce.number().int().min(min).max(max).default(fallback));
const price = z.preprocess(blankToUndefined, z.coerce.number().nonnegative().optional());
const flag = (fallback: boolean) =>
  z.preprocess(
    blankToUndefined,
    z
      .enum(["true", "false", "1", "0", "on", "off"])
      .default(fallback ? "true" : "false")
      .transform((value) => ["true", "1", "on"].includes(value)),
  );

const Env = z.object({
  LLM_PROVIDER: z.preprocess(
    blankToUndefined,
    z.enum(["anthropic", "openai", "google", "offline"]).default("offline"),
  ),
  LLM_MODEL: text,
  LLM_MAX_RETRIES: int(2, 0, 5),
  LLM_TIMEOUT_MS: int(60_000, 1_000, 600_000),
  LLM_MAX_TOKENS: int(16_000, 256, 128_000),
  AGENT_EFFORT: z.preprocess(
    blankToUndefined,
    z.enum(["low", "medium", "high", "xhigh", "max"]).optional(),
  ),
  ANTHROPIC_API_KEY: text,
  ANTHROPIC_API_URL: z.preprocess(
    blankToUndefined,
    z.url().default("https://api.anthropic.com"),
  ),
  ANTHROPIC_FALLBACKS: flag(true),
  OPENAI_API_KEY: text,
  GOOGLE_API_KEY: text,

  KNOWLEDGE_URL: z.preprocess(blankToUndefined, z.url().default("http://127.0.0.1:8100")),
  KNOWLEDGE_API_TOKEN: text,
  KNOWLEDGE_TIMEOUT_MS: int(45_000, 1_000, 300_000),

  AGENT_HOST: z.preprocess(blankToUndefined, z.string().default("127.0.0.1")),
  AGENT_PORT: int(8200, 1, 65_535),
  AGENT_MAX_STEPS: int(6, 1, 20),
  AGENT_MAX_TURNS: int(20, 1, 200),
  AGENT_MAX_MESSAGE_CHARS: int(2_000, 50, 20_000),
  RATE_LIMIT_PER_MINUTE: int(30, 1, 10_000),
  TOOL_TIMEOUT_MS: int(10_000, 500, 120_000),
  GITHUB_TOKEN: text,
  LOG_LEVEL: z.preprocess(
    blankToUndefined,
    z.enum(["debug", "info", "warn", "error", "silent"]).default("info"),
  ),

  LLM_PRICE_INPUT_PER_MTOK: price,
  LLM_PRICE_OUTPUT_PER_MTOK: price,
});

export type LlmProvider = z.infer<typeof Env>["LLM_PROVIDER"];

export interface Config {
  llm: {
    provider: LlmProvider;
    /** Vazio só no modo offline. */
    model: string;
    apiKey?: string;
    maxRetries: number;
    timeoutMs: number;
    maxTokens: number;
    effort?: "low" | "medium" | "high" | "xhigh" | "max";
    anthropicApiUrl: string;
    anthropicFallbacks: boolean;
    price?: { inputPerMTok: number; outputPerMTok: number };
  };
  knowledge: { url: string; token?: string; timeoutMs: number };
  server: { host: string; port: number; rateLimitPerMinute: number; logLevel: string };
  agent: { maxSteps: number; maxTurns: number; maxMessageChars: number };
  tools: { timeoutMs: number; githubToken?: string };
}

/** Só a Anthropic tem modelo padrão: os outros provedores exigem LLM_MODEL explícito. */
const DEFAULT_MODELS: Partial<Record<LlmProvider, string>> = { anthropic: "claude-opus-5-5" };

const KEY_NAMES = {
  anthropic: "ANTHROPIC_API_KEY",
  openai: "OPENAI_API_KEY",
  google: "GOOGLE_API_KEY",
} as const;

export class ConfigError extends Error {}

export function loadConfig(source: Record<string, string | undefined> = process.env): Config {
  const parsed = Env.safeParse(source);
  if (!parsed.success) {
    const issues = parsed.error.issues
      .map((issue) => `${issue.path.join(".")}: ${issue.message}`)
      .join("; ");
    throw new ConfigError(`Configuração inválida: ${issues}`);
  }
  const env = parsed.data;
  const provider = env.LLM_PROVIDER;

  let model = "";
  let apiKey: string | undefined;
  if (provider !== "offline") {
    const chosen = env.LLM_MODEL ?? DEFAULT_MODELS[provider];
    if (!chosen) throw new ConfigError(`LLM_MODEL é obrigatório com LLM_PROVIDER=${provider}`);
    model = chosen;
    const keyName = KEY_NAMES[provider];
    apiKey = env[keyName];
    if (!apiKey) throw new ConfigError(`${keyName} é obrigatória com LLM_PROVIDER=${provider}`);
  }

  const { LLM_PRICE_INPUT_PER_MTOK: priceIn, LLM_PRICE_OUTPUT_PER_MTOK: priceOut } = env;

  return {
    llm: {
      provider,
      model,
      apiKey,
      maxRetries: env.LLM_MAX_RETRIES,
      timeoutMs: env.LLM_TIMEOUT_MS,
      maxTokens: env.LLM_MAX_TOKENS,
      effort: env.AGENT_EFFORT,
      anthropicApiUrl: env.ANTHROPIC_API_URL,
      anthropicFallbacks: env.ANTHROPIC_FALLBACKS,
      price:
        priceIn !== undefined && priceOut !== undefined
          ? { inputPerMTok: priceIn, outputPerMTok: priceOut }
          : undefined,
    },
    knowledge: {
      url: env.KNOWLEDGE_URL.replace(/\/+$/, ""),
      token: env.KNOWLEDGE_API_TOKEN,
      timeoutMs: env.KNOWLEDGE_TIMEOUT_MS,
    },
    server: {
      host: env.AGENT_HOST,
      port: env.AGENT_PORT,
      rateLimitPerMinute: env.RATE_LIMIT_PER_MINUTE,
      logLevel: env.LOG_LEVEL,
    },
    agent: {
      maxSteps: env.AGENT_MAX_STEPS,
      maxTurns: env.AGENT_MAX_TURNS,
      maxMessageChars: env.AGENT_MAX_MESSAGE_CHARS,
    },
    tools: { timeoutMs: env.TOOL_TIMEOUT_MS, githubToken: env.GITHUB_TOKEN },
  };
}

/** Carrega o .env da raiz do repositório, compartilhado com o serviço Python. */
export function loadEnvFile(): void {
  if (process.env.BULA_SKIP_ENV_FILE) return;
  const path = fileURLToPath(new URL("../../.env", import.meta.url));
  if (existsSync(path)) process.loadEnvFile(path);
}
