/**
 * Modo offline (LLM_PROVIDER=offline): um substituto determinístico do modelo.
 *
 * Não é um modelo de linguagem e não redige resposta. Ele escolhe a tool por regra
 * simples e devolve os trechos encontrados, citados. Existe para o projeto rodar de
 * ponta a ponta sem chave de API (grafo, tools, guardrails, API e streaming de
 * verdade) e é anunciado no log da subida e no /health.
 */
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage, type BaseMessage } from "@langchain/core/messages";
import type { ChatResult } from "@langchain/core/outputs";

const GREETING = /^\s*(oi|olá|ola|hello|hi|hey|bom dia|boa tarde|boa noite)\b[\s!.,?]*$/i;
const PACKAGE_FACT = /\b(vers[aã]o|version|downloads?|latest|mais recente|[úu]ltima)\b/i;
const RELEASE_FACT = /\b(release|lan[çc]amento|changelog|o que mudou|what changed)\b/i;
const REPOSITORY = /\b([A-Za-z0-9][A-Za-z0-9-]*\/[A-Za-z0-9_.-]+)\b/;
// A palavra específica ganha da preposição: em "do pacote fastify" o nome é "fastify".
const PACKAGE_AFTER_NOUN =
  /\b(?:pacote|package|biblioteca|library)\s+(@?[a-z0-9][a-z0-9._-]*(?:\/[a-z0-9._-]+)?)/i;
const PACKAGE_AFTER_PREPOSITION =
  /\b(?:do|da|of|for)\s+(@?[a-z0-9][a-z0-9._-]*(?:\/[a-z0-9._-]+)?)/i;
// Sem a flag `m`: o `$` precisa ser o fim do texto, não o fim da primeira linha do trecho.
const NUMBERED_BLOCK =
  /(?<=^|\n)\[(\d+)\] (.+)\n(?:URL: .+\n)?([\s\S]*?)(?=\n\n\[\d+\] |\n<\/result>|$)/g;

export class OfflineChatModel extends BaseChatModel {
  _llmType(): string {
    return "bula-offline";
  }

  override bindTools(): this {
    return this;
  }

  _generate(messages: BaseMessage[]): Promise<ChatResult> {
    const message = respond(messages);
    return Promise.resolve({ generations: [{ text: message.text, message }] });
  }
}

export function respond(messages: BaseMessage[]): AIMessage {
  const lastHuman = messages.findLastIndex((message) => message.type === "human");
  const question = lastHuman >= 0 ? (messages[lastHuman]?.text ?? "") : "";
  const results = messages.slice(lastHuman + 1).filter((message) => message.type === "tool");

  if (results.length === 0) return firstMove(question);

  const found = results.flatMap((message) =>
    [...message.text.matchAll(NUMBERED_BLOCK)].map((match) => ({
      n: match[1] ?? "",
      title: match[2] ?? "",
      body: (match[3] ?? "").trim(),
    })),
  );
  if (found.length === 0) {
    return new AIMessage("Não encontrei nada sobre isso nas fontes consultadas.");
  }
  const lines = found.slice(0, 3).map((item) => {
    const excerpt = item.body.replace(/\s+/g, " ").slice(0, 280);
    return `- ${item.title} [${item.n}]: ${excerpt}${item.body.length > 280 ? "…" : ""}`;
  });
  return new AIMessage(
    `Modo offline, sem modelo de linguagem: estes são os trechos mais próximos da pergunta.\n\n${lines.join("\n")}`,
  );
}

function firstMove(question: string): AIMessage {
  if (GREETING.test(question)) {
    return new AIMessage(
      "Olá! Respondo perguntas sobre a documentação indexada e consulto versões no npm e releases no GitHub.",
    );
  }
  const repository = REPOSITORY.exec(question)?.[1];
  if (RELEASE_FACT.test(question) && repository) {
    return toolCall("github_latest_release", { repository });
  }
  const packageName = (PACKAGE_AFTER_NOUN.exec(question) ??
    PACKAGE_AFTER_PREPOSITION.exec(question))?.[1];
  if (PACKAGE_FACT.test(question) && packageName) {
    return toolCall("npm_package", { name: packageName.toLowerCase() });
  }
  return toolCall("search_docs", { question });
}

function toolCall(name: string, args: Record<string, unknown>): AIMessage {
  return new AIMessage({
    content: "",
    tool_calls: [{ id: `offline-${name}`, name, args, type: "tool_call" }],
  });
}
