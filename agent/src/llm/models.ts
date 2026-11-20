/**
 * Fábrica do modelo de chat. O resto do agente só conhece `BaseChatModel`.
 *
 * Notas da Anthropic, conferidas no `@langchain/anthropic` instalado:
 * - a URL base vai explícita, porque sem isso o adaptador herda `ANTHROPIC_BASE_URL`
 *   do ambiente e a chave iria para o host que estiver lá;
 * - nada de `temperature`, `topP` ou `topK`: os modelos atuais recusam;
 * - nada de `tool_choice` forçado: o Claude Opus 5.5 recusa, e o agente usa `auto`;
 * - `thinking` fica no padrão do modelo (adaptativo) e o esforço é opcional;
 * - o fallback de recusa do lado do servidor entra por padrão nos modelos que o aceitam.
 */
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { Config } from "../config.js";
import { OfflineChatModel } from "./offline.js";

export const FALLBACK_BETA = "server-side-fallback-2026-07-01";

const FALLBACK_MODELS = [
  "claude-fable-5-1",
  "claude-fable-5-2",
  "claude-opus-5",
  "claude-sonnet-5-5",
];

export async function createChatModel(llm: Config["llm"]): Promise<BaseChatModel> {
  switch (llm.provider) {
    case "anthropic": {
      const { ChatAnthropic } = await import("@langchain/anthropic");
      const fallbacks =
        llm.anthropicFallbacks && FALLBACK_MODELS.some((prefix) => llm.model.startsWith(prefix));
      return new ChatAnthropic({
        model: llm.model,
        apiKey: llm.apiKey,
        anthropicApiUrl: llm.anthropicApiUrl,
        maxRetries: llm.maxRetries,
        maxTokens: llm.maxTokens,
        clientOptions: { timeout: llm.timeoutMs },
        ...(llm.effort ? { outputConfig: { effort: llm.effort } } : {}),
        ...(fallbacks
          ? { betas: [FALLBACK_BETA], invocationKwargs: { fallbacks: "default" } }
          : {}),
      });
    }
    case "openai": {
      const { ChatOpenAI } = await import("@langchain/openai");
      return new ChatOpenAI({
        model: llm.model,
        apiKey: llm.apiKey,
        maxRetries: llm.maxRetries,
        timeout: llm.timeoutMs,
      });
    }
    case "google": {
      const { ChatGoogleGenerativeAI } = await import("@langchain/google-genai");
      return new ChatGoogleGenerativeAI({
        model: llm.model,
        apiKey: llm.apiKey,
        maxRetries: llm.maxRetries,
      });
    }
    case "offline":
      return new OfflineChatModel({});
  }
}
