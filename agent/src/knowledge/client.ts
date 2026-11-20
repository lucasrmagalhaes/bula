/**
 * Cliente do serviço de conhecimento (Python).
 *
 * Os schemas espelham `knowledge/src/bula_knowledge/schemas.py`. Toda resposta é
 * validada: se o contrato mudar do outro lado, o erro aparece aqui, tipado, e não
 * como `undefined` no meio do agente.
 */
import { z } from "zod";

export const ChunkSchema = z.object({
  id: z.number().int(),
  source: z.string(),
  title: z.string(),
  heading: z.string(),
  url: z.string(),
  content: z.string(),
  score: z.number(),
});

export const ContextResponseSchema = z.object({
  status: z.enum(["ok", "no_context"]),
  question: z.string(),
  queries: z.array(z.string()),
  attempts: z.number().int(),
  chunks: z.array(ChunkSchema),
  mode: z.enum(["llm", "offline"]),
  degraded: z.boolean().default(false),
  usage: z.object({
    llm_calls: z.number().int(),
    input_tokens: z.number().int(),
    output_tokens: z.number().int(),
  }),
  timings_ms: z.record(z.string(), z.number()),
  trace_id: z.string().nullish(),
});

export const SourcesResponseSchema = z.object({
  embedding_model: z.string(),
  sources: z.array(
    z.object({
      id: z.string(),
      title: z.string(),
      repo: z.string(),
      ref: z.string(),
      license: z.string(),
      documents: z.number().int(),
      chunks: z.number().int(),
    }),
  ),
});

export type Chunk = z.infer<typeof ChunkSchema>;
export type ContextResponse = z.infer<typeof ContextResponseSchema>;
export type SourcesResponse = z.infer<typeof SourcesResponseSchema>;

export type KnowledgeErrorKind = "unavailable" | "timeout" | "http" | "contract";

export class KnowledgeError extends Error {
  constructor(
    readonly kind: KnowledgeErrorKind,
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "KnowledgeError";
  }
}

export interface KnowledgeClientOptions {
  baseUrl: string;
  token?: string;
  timeoutMs: number;
  fetch?: typeof fetch;
}

export interface KnowledgeApi {
  context(
    question: string,
    options?: { sources?: string[]; requestId?: string },
  ): Promise<ContextResponse>;
  sources(): Promise<SourcesResponse>;
}

export class KnowledgeClient implements KnowledgeApi {
  private readonly fetchImpl: typeof fetch;

  constructor(private readonly options: KnowledgeClientOptions) {
    this.fetchImpl = options.fetch ?? fetch;
  }

  context(
    question: string,
    options: { sources?: string[]; requestId?: string } = {},
  ): Promise<ContextResponse> {
    return this.request(
      "/v1/context",
      ContextResponseSchema,
      { question, sources: options.sources ?? null },
      options.requestId,
    );
  }

  sources(): Promise<SourcesResponse> {
    return this.request("/v1/sources", SourcesResponseSchema);
  }

  private async request<S extends z.ZodType>(
    path: string,
    schema: S,
    body?: unknown,
    requestId?: string,
  ): Promise<z.infer<S>> {
    const headers: Record<string, string> = { accept: "application/json" };
    if (body !== undefined) headers["content-type"] = "application/json";
    if (this.options.token) headers.authorization = `Bearer ${this.options.token}`;
    if (requestId) headers["x-request-id"] = requestId;

    let response: Response;
    try {
      response = await this.fetchImpl(`${this.options.baseUrl}${path}`, {
        method: body === undefined ? "GET" : "POST",
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: AbortSignal.timeout(this.options.timeoutMs),
      });
    } catch (error) {
      if (error instanceof Error && error.name === "TimeoutError") {
        throw new KnowledgeError(
          "timeout",
          `serviço de conhecimento não respondeu em ${this.options.timeoutMs} ms`,
        );
      }
      throw new KnowledgeError("unavailable", "serviço de conhecimento fora do ar");
    }

    if (!response.ok) {
      throw new KnowledgeError(
        "http",
        `serviço de conhecimento respondeu ${response.status}`,
        response.status,
      );
    }

    const parsed = schema.safeParse(await response.json().catch(() => null));
    if (!parsed.success) {
      throw new KnowledgeError("contract", `resposta de ${path} fora do contrato`);
    }
    return parsed.data;
  }
}
