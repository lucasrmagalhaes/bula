import { z } from "zod";
import { KnowledgeError, type KnowledgeApi } from "../knowledge/client.js";
import { ToolError, type AgentTool } from "./types.js";

const schema = z.object({
  question: z
    .string()
    .describe(
      "The question to look up, self-contained: include the library name and any exact " +
        "identifier (function, option, error code). Any language.",
    ),
});

/** Título do documento mais o caminho da seção, sem repetir o título quando ele é o H1. */
function fullHeading(title: string, heading: string): string {
  return heading === title || heading.startsWith(`${title} › `) ? heading : `${title} › ${heading}`;
}

/** Consulta o workflow de RAG do serviço Python e devolve os trechos que ele aprovou. */
export function searchDocsTool(knowledge: KnowledgeApi): AgentTool<typeof schema> {
  return {
    name: "search_docs",
    description:
      "Searches the indexed library documentation and returns the passages that help " +
      "answer the question, each with its source. Use it for anything about how a " +
      "library works: APIs, options, behaviour, errors.",
    schema,
    async run({ question }, context) {
      let result;
      try {
        result = await knowledge.context(question, { requestId: context.requestId });
      } catch (error) {
        if (error instanceof KnowledgeError) {
          throw new ToolError(
            error.kind === "timeout" ? "timeout" : "unavailable",
            "a busca na documentação está indisponível",
          );
        }
        throw error;
      }
      const usage = {
        inputTokens: result.usage.input_tokens,
        outputTokens: result.usage.output_tokens,
        llmCalls: result.usage.llm_calls,
      };
      if (result.status === "no_context") {
        return {
          items: [],
          note: "No relevant passage was found in the indexed documentation for this question.",
          usage,
        };
      }
      return {
        items: result.chunks.map((chunk) => ({
          kind: "docs" as const,
          title: fullHeading(chunk.title, chunk.heading),
          url: chunk.url,
          body: chunk.content,
        })),
        usage,
      };
    },
  };
}
