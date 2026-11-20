import type { z } from "zod";
import { ToolError } from "./types.js";

export interface HttpOptions {
  timeoutMs: number;
  headers?: Record<string, string>;
  fetch?: typeof fetch;
}

/**
 * GET de JSON para as tools: tempo limite, erro tipado e resposta validada.
 * Devolve `null` em 404, que para estas APIs significa "não existe" e não é falha.
 */
export async function getJson<S extends z.ZodType>(
  url: string,
  schema: S,
  options: HttpOptions,
): Promise<z.infer<S> | null> {
  const fetchImpl = options.fetch ?? fetch;
  const host = new URL(url).host;
  let response: Response;
  try {
    response = await fetchImpl(url, {
      headers: { accept: "application/json", "user-agent": "bula-agent", ...options.headers },
      signal: AbortSignal.timeout(options.timeoutMs),
    });
  } catch (error) {
    if (error instanceof Error && error.name === "TimeoutError") {
      throw new ToolError("timeout", `${host} não respondeu em ${options.timeoutMs} ms`);
    }
    throw new ToolError("network", `não foi possível conectar a ${host}`);
  }

  if (response.status === 404) return null;
  if (response.status === 429 || response.headers.get("x-ratelimit-remaining") === "0") {
    throw new ToolError("rate_limit", `limite de requisições de ${host} atingido`);
  }
  if (!response.ok) {
    throw new ToolError("upstream", `${host} respondeu ${response.status}`);
  }

  const parsed = schema.safeParse(await response.json().catch(() => null));
  if (!parsed.success) {
    throw new ToolError("shape", `resposta de ${host} em formato inesperado`);
  }
  return parsed.data;
}
