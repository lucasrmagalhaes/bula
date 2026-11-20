/** Dublês de teste. Ficam fora de `src`: nada daqui roda em produção. */
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import {
  BaseChatModel,
  type BindToolsInput,
} from "@langchain/core/language_models/chat_models";
import { AIMessage, AIMessageChunk, type BaseMessage } from "@langchain/core/messages";
import { ChatGenerationChunk, type ChatResult } from "@langchain/core/outputs";
import type {
  ContextResponse,
  KnowledgeApi,
  SourcesResponse,
} from "../src/knowledge/client.js";

export type Step = (messages: BaseMessage[]) => AIMessage;

/** Modelo roteirizado: a n-ésima chamada devolve o n-ésimo passo; o último se repete. */
export class ScriptedChatModel extends BaseChatModel {
  readonly calls: BaseMessage[][] = [];
  boundTools: BindToolsInput[] = [];

  constructor(private readonly steps: Step[]) {
    super({});
  }

  _llmType(): string {
    return "scripted";
  }

  override bindTools(tools: BindToolsInput[]): this {
    this.boundTools = tools;
    return this;
  }

  private next(messages: BaseMessage[]): AIMessage {
    this.calls.push(messages);
    const step = this.steps[Math.min(this.calls.length, this.steps.length) - 1];
    if (!step) throw new Error("roteiro vazio");
    return step(messages);
  }

  _generate(messages: BaseMessage[]): Promise<ChatResult> {
    const message = this.next(messages);
    return Promise.resolve({ generations: [{ text: message.text, message }] });
  }

  /** Com streaming, o texto sai em dois pedaços, como um provedor real faria. */
  override async *_streamResponseChunks(
    messages: BaseMessage[],
    _options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun,
  ): AsyncGenerator<ChatGenerationChunk> {
    const message = this.next(messages);
    const text = message.text;
    const half = Math.ceil(text.length / 2);
    const pieces = text.length > 1 ? [text.slice(0, half), text.slice(half)] : [text];
    for (const [index, piece] of pieces.entries()) {
      const last = index === pieces.length - 1;
      const chunk = new ChatGenerationChunk({
        text: piece,
        message: new AIMessageChunk({
          content: piece,
          tool_call_chunks: last
            ? (message.tool_calls ?? []).map((call, position) => ({
                id: call.id,
                name: call.name,
                args: JSON.stringify(call.args),
                index: position,
                type: "tool_call_chunk" as const,
              }))
            : [],
          usage_metadata: last ? message.usage_metadata : undefined,
        }),
      });
      yield chunk;
      // É o provedor quem avisa cada pedaço; sem isto ninguém vê o token antes do fim.
      await runManager?.handleLLMNewToken(piece, undefined, undefined, undefined, undefined, {
        chunk,
      });
    }
  }
}

export function say(text: string, tokens = { input: 100, output: 20 }): Step {
  return () =>
    new AIMessage({
      content: text,
      usage_metadata: {
        input_tokens: tokens.input,
        output_tokens: tokens.output,
        total_tokens: tokens.input + tokens.output,
      },
    });
}

let callCounter = 0;

export function call(...calls: [name: string, args: Record<string, unknown>][]): Step {
  return () =>
    new AIMessage({
      content: "",
      tool_calls: calls.map(([name, args]) => ({
        id: `call_${++callCounter}`,
        name,
        args,
        type: "tool_call" as const,
      })),
      usage_metadata: { input_tokens: 50, output_tokens: 10, total_tokens: 60 },
    });
}

export function contextOf(
  chunks: { title: string; heading: string; url: string; content: string }[],
): ContextResponse {
  return {
    status: chunks.length > 0 ? "ok" : "no_context",
    question: "",
    queries: ["q"],
    attempts: 1,
    chunks: chunks.map((chunk, index) => ({ id: index + 1, source: "fastify", score: 0.03, ...chunk })),
    mode: "llm",
    degraded: false,
    usage: { llm_calls: 2, input_tokens: 300, output_tokens: 40 },
    timings_ms: { total: 12 },
    trace_id: null,
  };
}

export const HOOKS_CHUNK = {
  title: "Hooks",
  heading: "Hooks › onRequest",
  url: "https://github.com/fastify/fastify/blob/abc/docs/Reference/Hooks.md#onrequest",
  content: "The onRequest hook runs before the request is parsed.",
};

export class FakeKnowledge implements KnowledgeApi {
  readonly questions: string[] = [];

  constructor(
    private readonly reply: (question: string) => ContextResponse | Promise<ContextResponse>,
  ) {}

  async context(question: string): Promise<ContextResponse> {
    this.questions.push(question);
    return this.reply(question);
  }

  sources(): Promise<SourcesResponse> {
    return Promise.resolve({ embedding_model: "fake", sources: [] });
  }
}

/** `fetch` de mentira: responde pela primeira rota cujo trecho aparece na URL. */
export function fakeFetch(
  routes: Record<string, { status?: number; body?: unknown; headers?: Record<string, string> } | Error>,
): typeof fetch & { urls: string[]; headers: Record<string, string>[] } {
  const urls: string[] = [];
  const headers: Record<string, string>[] = [];
  const impl = (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
    const url = input instanceof Request ? input.url : input.toString();
    urls.push(url);
    headers.push(Object.fromEntries(new Headers(init?.headers).entries()));
    const match = Object.entries(routes).find(([fragment]) => url.includes(fragment));
    if (!match) return Promise.resolve(new Response("not found", { status: 404 }));
    const [, route] = match;
    if (route instanceof Error) return Promise.reject(route);
    return Promise.resolve(
      new Response(JSON.stringify(route.body ?? {}), {
        status: route.status ?? 200,
        headers: { "content-type": "application/json", ...route.headers },
      }),
    );
  };
  return Object.assign(impl, { urls, headers });
}

export function timeoutError(): Error {
  return Object.assign(new Error("timed out"), { name: "TimeoutError" });
}
