import type { z } from "zod";

/** Um resultado citável: vira uma fonte numerada [n] na conversa. */
export interface SourceItem {
  kind: "docs" | "npm" | "github";
  title: string;
  url: string;
  body: string;
}

export interface ToolResult {
  /** Fontes citáveis. Vazio quando a tool não achou nada. */
  items: SourceItem[];
  /** Texto para o modelo quando não há fontes, como "nada encontrado". */
  note?: string;
  /** Tokens gastos por LLM dentro da tool (o workflow de RAG usa um). */
  usage?: { inputTokens: number; outputTokens: number; llmCalls: number };
}

export interface ToolContext {
  requestId: string;
}

export interface AgentTool<S extends z.ZodType = z.ZodType> {
  name: string;
  description: string;
  schema: S;
  run(args: z.infer<S>, context: ToolContext): Promise<ToolResult>;
}

/** Erro esperado de uma tool. A mensagem vai para o modelo, então não carrega detalhe interno. */
export class ToolError extends Error {
  constructor(
    readonly kind: "timeout" | "network" | "upstream" | "rate_limit" | "shape" | "unavailable",
    message: string,
  ) {
    super(message);
    this.name = "ToolError";
  }
}
