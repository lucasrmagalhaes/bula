/**
 * Executor do agente: uma conversa por `threadId`, com memória no checkpointer.
 *
 * Controles de custo que ficam aqui, fora do modelo:
 * - teto de turnos por conversa (o histórico só cresce, então a conversa tem fim);
 * - teto de conversas em memória, com descarte da menos usada;
 * - um turno por vez em cada conversa, para o histórico não se intercalar.
 */
import { randomUUID } from "node:crypto";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage, AIMessageChunk, HumanMessage } from "@langchain/core/messages";
import { MemorySaver } from "@langchain/langgraph";
import type { AgentTool } from "../tools/types.js";
import { buildAgentGraph, type Usage } from "./graph.js";
import type { Outcome, Verdict } from "./guardrails.js";

export interface Citation {
  n: number;
  kind: "docs" | "npm" | "github";
  title: string;
  url: string;
}

export interface ChatResult {
  threadId: string;
  requestId: string;
  answer: string;
  outcome: Outcome;
  citations: Citation[];
  steps: number;
  usage: Usage & { costUsd?: number };
  tookMs: number;
}

export type AgentEvent =
  | { type: "step"; node: string; tools?: string[] }
  | { type: "token"; text: string }
  | { type: "done"; result: ChatResult };

export interface AgentOptions {
  model: BaseChatModel;
  tools: AgentTool[];
  systemPrompt: string;
  maxSteps: number;
  maxTurns: number;
  maxThreads?: number;
  price?: { inputPerMTok: number; outputPerMTok: number };
  onToolFailure?: (tool: string, error: unknown, requestId: string) => void;
}

export interface AskOptions {
  threadId?: string;
  requestId?: string;
  signal?: AbortSignal;
}

export class ThreadFullError extends Error {
  constructor(readonly maxTurns: number) {
    super(`a conversa chegou ao limite de ${maxTurns} perguntas; comece uma nova`);
    this.name = "ThreadFullError";
  }
}

export class Agent {
  private readonly checkpointer = new MemorySaver();
  private readonly graph;
  private readonly locks = new Map<string, Promise<void>>();
  /** Ordem de inserção = ordem de uso; a primeira chave é a conversa menos recente. */
  private readonly threads = new Set<string>();
  private readonly maxThreads: number;

  constructor(private readonly options: AgentOptions) {
    this.maxThreads = options.maxThreads ?? 500;
    this.graph = buildAgentGraph({
      model: options.model,
      tools: options.tools,
      systemPrompt: options.systemPrompt,
      maxSteps: options.maxSteps,
      checkpointer: this.checkpointer,
      onToolFailure: options.onToolFailure,
    });
  }

  async ask(message: string, options: AskOptions = {}): Promise<ChatResult> {
    let result: ChatResult | undefined;
    for await (const event of this.stream(message, options)) {
      if (event.type === "done") result = event.result;
    }
    if (!result) throw new Error("o agente terminou sem resposta");
    return result;
  }

  async *stream(message: string, options: AskOptions = {}): AsyncGenerator<AgentEvent> {
    const threadId = options.threadId ?? randomUUID();
    const requestId = options.requestId ?? randomUUID();
    const release = await this.acquire(threadId);
    try {
      const config = {
        configurable: { thread_id: threadId, requestId },
        // Teto do LangGraph por cima do contador de passos do próprio estado.
        recursionLimit: 2 * this.options.maxSteps + 6,
        signal: options.signal,
      };
      const before = await this.graph.getState(config);
      const history = (before.values as { messages?: unknown[] }).messages ?? [];
      const turns = history.filter((item) => HumanMessage.isInstance(item)).length;
      if (turns >= this.options.maxTurns) throw new ThreadFullError(this.options.maxTurns);
      await this.touch(threadId);

      const started = performance.now();
      const stream = await this.graph.stream(
        { messages: [new HumanMessage(message)] },
        { ...config, streamMode: ["updates", "messages"] },
      );
      for await (const [mode, payload] of stream) {
        if (mode === "messages") {
          const [chunk, metadata] = payload as [unknown, { langgraph_node?: string }];
          if (metadata.langgraph_node === "agent" && AIMessageChunk.isInstance(chunk)) {
            if (chunk.text) yield { type: "token", text: chunk.text };
          }
          continue;
        }
        for (const [node, update] of Object.entries(payload as Record<string, unknown>)) {
          if (node === "begin") continue;
          yield { type: "step", node, tools: requestedTools(node, update) };
        }
      }

      const state = (await this.graph.getState(config)).values as {
        final: Verdict | null;
        turn: { steps: number; usage: Usage };
      };
      if (!state.final) throw new Error("o grafo terminou sem veredito");
      yield {
        type: "done",
        result: {
          threadId,
          requestId,
          answer: state.final.answer,
          outcome: state.final.outcome,
          citations: state.final.citations.map(({ n, kind, title, url }) => ({
            n,
            kind,
            title,
            url,
          })),
          steps: state.turn.steps,
          usage: { ...state.turn.usage, costUsd: this.cost(state.turn.usage) },
          tookMs: Math.round(performance.now() - started),
        },
      };
    } finally {
      release();
    }
  }

  private cost(usage: Usage): number | undefined {
    const price = this.options.price;
    if (!price) return undefined;
    const usd =
      (usage.inputTokens * price.inputPerMTok + usage.outputTokens * price.outputPerMTok) / 1e6;
    return Math.round(usd * 1e6) / 1e6;
  }

  /** Fila por conversa: devolve a função que libera a vez. */
  private async acquire(threadId: string): Promise<() => void> {
    const previous = this.locks.get(threadId) ?? Promise.resolve();
    let release!: () => void;
    const current = new Promise<void>((resolve) => {
      release = resolve;
    });
    const tail = previous.then(() => current);
    this.locks.set(threadId, tail);
    await previous;
    return () => {
      release();
      if (this.locks.get(threadId) === tail) this.locks.delete(threadId);
    };
  }

  private async touch(threadId: string): Promise<void> {
    this.threads.delete(threadId);
    this.threads.add(threadId);
    while (this.threads.size > this.maxThreads) {
      const oldest = this.threads.values().next().value;
      if (oldest === undefined) break;
      this.threads.delete(oldest);
      await this.checkpointer.deleteThread(oldest);
    }
  }
}

function requestedTools(node: string, update: unknown): string[] | undefined {
  if (node !== "agent") return undefined;
  const messages = (update as { messages?: unknown[] } | null)?.messages ?? [];
  const names = messages.flatMap((message) =>
    AIMessage.isInstance(message) ? (message.tool_calls ?? []).map((call) => call.name) : [],
  );
  return names.length > 0 ? names : undefined;
}
