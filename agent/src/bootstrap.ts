/** Monta o agente a partir da configuração. Usado pelo servidor e pela CLI. */
import { Agent } from "./agent/agent.js";
import { buildSystemPrompt } from "./agent/prompt.js";
import type { Config } from "./config.js";
import { KnowledgeClient, KnowledgeError, type KnowledgeApi } from "./knowledge/client.js";
import { createChatModel } from "./llm/models.js";
import { buildTools } from "./tools/index.js";

export interface Logger {
  info(details: object, message: string): void;
  warn(details: object, message: string): void;
}

export interface Runtime {
  agent: Agent;
  knowledge: KnowledgeApi;
  indexed: string[];
}

/**
 * Lê o que está indexado para pôr no prompt de sistema.
 *
 * O prompt não pode mudar depois da subida, então vale esperar o serviço de conhecimento
 * quando os dois sobem juntos. Só "fora do ar" e "tempo esgotado" são tentados de novo:
 * token recusado ou contrato quebrado não se resolvem esperando.
 */
export async function loadIndexed(
  knowledge: KnowledgeApi,
  options: { waitMs: number; intervalMs?: number; onGiveUp?: (kind: string) => void },
): Promise<string[]> {
  const deadline = Date.now() + options.waitMs;
  for (;;) {
    try {
      const { sources } = await knowledge.sources();
      return sources.map((source) => `${source.title} ${source.ref}`);
    } catch (error) {
      if (!(error instanceof KnowledgeError)) throw error;
      const transient = error.kind === "unavailable" || error.kind === "timeout";
      if (!transient || Date.now() >= deadline) {
        options.onGiveUp?.(error.kind);
        return [];
      }
      await new Promise((resolve) => setTimeout(resolve, options.intervalMs ?? 1_000));
    }
  }
}

export async function createRuntime(
  config: Config,
  logger: Logger,
  options: { waitForKnowledgeMs?: number } = {},
): Promise<Runtime> {
  const knowledge = new KnowledgeClient({
    baseUrl: config.knowledge.url,
    token: config.knowledge.token,
    timeoutMs: config.knowledge.timeoutMs,
  });

  const indexed = await loadIndexed(knowledge, {
    waitMs: options.waitForKnowledgeMs ?? 20_000,
    onGiveUp: (kind) => {
      logger.warn(
        { kind, url: config.knowledge.url },
        "serviço de conhecimento indisponível na subida; o agente segue sem a lista do índice",
      );
    },
  });

  if (config.llm.provider === "offline") {
    logger.warn(
      {},
      "LLM_PROVIDER=offline: sem modelo de linguagem; o agente devolve os trechos encontrados sem redigir resposta",
    );
  }

  const agent = new Agent({
    model: await createChatModel(config.llm),
    tools: buildTools({
      knowledge,
      timeoutMs: config.tools.timeoutMs,
      githubToken: config.tools.githubToken,
    }),
    systemPrompt: buildSystemPrompt(indexed),
    maxSteps: config.agent.maxSteps,
    maxTurns: config.agent.maxTurns,
    price: config.llm.price,
    onToolFailure: (tool, error, requestId) => {
      logger.warn(
        { tool, requestId, error: error instanceof Error ? error.message : String(error) },
        "tool falhou",
      );
    },
  });
  return { agent, knowledge, indexed };
}
