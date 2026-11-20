/**
 * Grafo do agente em LangGraph.
 *
 *   START -> begin -> agent -+-> tools -> agent ...
 *                            +-> finish -> END     (guardrail de saída)
 *                            +-> limit  -> END     (teto de passos)
 *
 * É um agente: o modelo escolhe as tools e quantas vezes consultar. O que não fica
 * com ele está em código: o teto de passos, a numeração das fontes, o tratamento de
 * erro de tool e a decisão de liberar ou reter a resposta.
 *
 * O histórico em `messages` só recebe acréscimos. A resposta que o usuário vê fica
 * em `final`, separada, para que o guardrail possa trocá-la sem reescrever uma
 * mensagem já enviada ao modelo.
 */
import { randomUUID } from "node:crypto";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage, SystemMessage, ToolMessage } from "@langchain/core/messages";
import type { RunnableConfig } from "@langchain/core/runnables";
import {
  Annotation,
  END,
  MessagesAnnotation,
  START,
  StateGraph,
  type BaseCheckpointSaver,
} from "@langchain/langgraph";
import { ToolError, type AgentTool, type SourceItem } from "../tools/types.js";
import { fixed, judge, type Source, type Verdict } from "./guardrails.js";

export interface Usage {
  inputTokens: number;
  outputTokens: number;
  llmCalls: number;
}

export interface TurnState {
  requestId: string;
  steps: number;
  toolCalls: number;
  toolErrors: number;
  sourceStart: number;
  usage: Usage;
}

const emptyTurn = (): TurnState => ({
  requestId: "",
  steps: 0,
  toolCalls: 0,
  toolErrors: 0,
  sourceStart: 0,
  usage: { inputTokens: 0, outputTokens: 0, llmCalls: 0 },
});

export const AgentState = Annotation.Root({
  ...MessagesAnnotation.spec,
  sources: Annotation<Source[]>({
    reducer: (current, update) => [...current, ...update],
    default: () => [],
  }),
  turn: Annotation<TurnState>({ reducer: (_current, update) => update, default: emptyTurn }),
  final: Annotation<Verdict | null>({ reducer: (_current, update) => update, default: () => null }),
});

type State = typeof AgentState.State;
type Update = typeof AgentState.Update;

export interface GraphDeps {
  model: BaseChatModel;
  tools: AgentTool[];
  systemPrompt: string;
  maxSteps: number;
  checkpointer: BaseCheckpointSaver;
  onToolFailure?: (tool: string, error: unknown, requestId: string) => void;
}

export function buildAgentGraph(deps: GraphDeps) {
  const definitions = deps.tools.map(({ name, description, schema }) => ({
    name,
    description,
    schema,
  }));
  const model = deps.model.bindTools ? deps.model.bindTools(definitions) : deps.model;
  const toolsByName = new Map(deps.tools.map((tool) => [tool.name, tool]));
  const system = new SystemMessage(deps.systemPrompt);

  const begin = (state: State, config: RunnableConfig): Update => ({
    turn: {
      ...emptyTurn(),
      requestId: (config.configurable?.requestId as string | undefined) ?? randomUUID(),
      sourceStart: state.sources.length,
    },
    final: null,
  });

  const agent = async (state: State, config: RunnableConfig): Promise<Update> => {
    const response = await model.invoke([system, ...state.messages], config);
    const usage = AIMessage.isInstance(response) ? response.usage_metadata : undefined;
    return {
      messages: [response],
      turn: {
        ...state.turn,
        steps: state.turn.steps + 1,
        usage: addUsage(state.turn.usage, {
          inputTokens: usage?.input_tokens ?? 0,
          outputTokens: usage?.output_tokens ?? 0,
          llmCalls: 1,
        }),
      },
    };
  };

  const tools = async (state: State): Promise<Update> => {
    const calls = pendingToolCalls(state);
    // As tools são leituras independentes: rodam em paralelo e são numeradas na ordem pedida.
    const outcomes = await Promise.all(
      calls.map((call) => runTool(toolsByName.get(call.name), call, state.turn.requestId)),
    );

    const numbered = new Map(state.sources.map((source) => [source.key, source.n]));
    const added: Source[] = [];
    let usage = state.turn.usage;
    let errors = 0;
    const messages = calls.map((call, index) => {
      const outcome = outcomes[index];
      if (!outcome || outcome.error !== undefined) {
        errors += 1;
        if (outcome?.cause !== undefined) {
          deps.onToolFailure?.(call.name, outcome.cause, state.turn.requestId);
        }
        return toolMessage(call, `Tool failed: ${outcome?.error ?? "unknown error"}`, "error");
      }
      if (outcome.usage) usage = addUsage(usage, outcome.usage);
      if (outcome.items.length === 0) {
        return toolMessage(call, outcome.note ?? "No results.", "success");
      }
      const blocks = outcome.items.map((item) => {
        const key = sourceKey(item);
        let n = numbered.get(key);
        if (n === undefined) {
          n = state.sources.length + added.length + 1;
          numbered.set(key, n);
          added.push({ n, kind: item.kind, title: item.title, url: item.url, key });
        }
        return `[${n}] ${item.title}\nURL: ${item.url}\n${item.body}`;
      });
      return toolMessage(
        call,
        `<result tool="${call.name}">\n${blocks.join("\n\n")}\n</result>`,
        "success",
      );
    });

    return {
      messages,
      sources: added,
      turn: {
        ...state.turn,
        toolCalls: state.turn.toolCalls + calls.length,
        toolErrors: state.turn.toolErrors + errors,
        usage,
      },
    };
  };

  const finish = (state: State): Update => {
    const last = lastAiMessage(state);
    if (last?.response_metadata.stop_reason === "refusal") return { final: fixed("refused") };
    return {
      final: judge({
        text: last?.text ?? "",
        sources: state.sources,
        turnSourceStart: state.turn.sourceStart,
        toolCalls: state.turn.toolCalls,
        toolErrors: state.turn.toolErrors,
      }),
    };
  };

  const limit = (state: State): Update => ({
    // Toda chamada de tool pendente recebe um resultado: sem isso o histórico fica
    // inválido para o próximo turno da conversa.
    messages: pendingToolCalls(state).map((call) =>
      toolMessage(call, "Not executed: the step limit for this question was reached.", "error"),
    ),
    final: fixed("step_limit"),
  });

  const route = (state: State): "tools" | "finish" | "limit" => {
    if (pendingToolCalls(state).length === 0) return "finish";
    return state.turn.steps >= deps.maxSteps ? "limit" : "tools";
  };

  return new StateGraph(AgentState)
    .addNode("begin", begin)
    .addNode("agent", agent)
    .addNode("tools", tools)
    .addNode("finish", finish)
    .addNode("limit", limit)
    .addEdge(START, "begin")
    .addEdge("begin", "agent")
    .addConditionalEdges("agent", route, ["tools", "finish", "limit"])
    .addEdge("tools", "agent")
    .addEdge("finish", END)
    .addEdge("limit", END)
    .compile({ checkpointer: deps.checkpointer });
}

interface ToolCall {
  id?: string;
  name: string;
  args: Record<string, unknown>;
}

type ToolOutcome =
  | { error: string; cause?: unknown }
  | { error?: undefined; items: SourceItem[]; note?: string; usage?: Usage };

async function runTool(
  tool: AgentTool | undefined,
  call: ToolCall,
  requestId: string,
): Promise<ToolOutcome> {
  if (!tool) return { error: `unknown tool "${call.name}"` };
  const args = tool.schema.safeParse(call.args);
  if (!args.success) {
    const issues = args.error.issues.map((issue) => issue.message).join("; ");
    return { error: `invalid arguments (${issues})` };
  }
  try {
    return await tool.run(args.data, { requestId });
  } catch (error) {
    // Erro esperado tem mensagem escrita para o modelo. O inesperado não vaza detalhe.
    if (error instanceof ToolError) return { error: error.message, cause: error };
    return { error: "unexpected error", cause: error };
  }
}

function pendingToolCalls(state: State): ToolCall[] {
  const last = state.messages.at(-1);
  return AIMessage.isInstance(last) ? (last.tool_calls ?? []) : [];
}

function lastAiMessage(state: State): AIMessage | undefined {
  return state.messages.findLast((message) => AIMessage.isInstance(message));
}

function toolMessage(call: ToolCall, content: string, status: "success" | "error"): ToolMessage {
  return new ToolMessage({ content, tool_call_id: call.id ?? "", name: call.name, status });
}

function sourceKey(item: SourceItem): string {
  return `${item.url}|${item.body.length}|${item.body.slice(0, 64)}`;
}

function addUsage(left: Usage, right: Usage): Usage {
  return {
    inputTokens: left.inputTokens + right.inputTokens,
    outputTokens: left.outputTokens + right.outputTokens,
    llmCalls: left.llmCalls + right.llmCalls,
  };
}
