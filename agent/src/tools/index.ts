import type { KnowledgeApi } from "../knowledge/client.js";
import { searchDocsTool } from "./docs.js";
import { githubReleaseTool } from "./github.js";
import type { HttpOptions } from "./http.js";
import { npmPackageTool } from "./npm.js";
import type { AgentTool } from "./types.js";

export interface ToolsOptions {
  knowledge: KnowledgeApi;
  timeoutMs: number;
  githubToken?: string;
  fetch?: typeof fetch;
}

/** Todas as tools são leituras: nenhuma altera estado fora do processo. */
export function buildTools(options: ToolsOptions): AgentTool[] {
  const http: HttpOptions = { timeoutMs: options.timeoutMs, fetch: options.fetch };
  return [
    searchDocsTool(options.knowledge),
    npmPackageTool(http),
    githubReleaseTool(http, options.githubToken),
  ] as AgentTool[];
}
