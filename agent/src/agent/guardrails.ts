/**
 * Guardrail de saída, em código. O modelo redige; quem decide se a resposta sai é isto.
 *
 * Regras:
 * 1. Resposta com citação válida sai, sem as citações que apontam para fonte inexistente.
 * 2. Houve consulta e nenhuma fonte voltou: sai a mensagem fixa de "não encontrei"
 *    (ou a de falha, se alguma tool falhou). O texto do modelo é descartado.
 * 3. Houve fontes e a resposta não cita nenhuma: fica retida.
 * 4. Sem consulta nenhuma, só passa resposta curta (saudação, pedido de esclarecimento,
 *    aviso de que a pergunta está fora do escopo).
 *    Texto longo sem fonte é o modelo respondendo de memória, e fica retido.
 */
export type Outcome =
  | "answered"
  | "direct"
  | "not_found"
  | "tool_error"
  | "blocked"
  | "refused"
  | "step_limit";

export interface Source {
  n: number;
  kind: "docs" | "npm" | "github";
  title: string;
  url: string;
  /** Identidade do conteúdo, para não numerar duas vezes o mesmo trecho. */
  key: string;
}

export interface Verdict {
  outcome: Outcome;
  answer: string;
  citations: Source[];
}

export const MESSAGES = {
  not_found:
    "Não encontrei isso na documentação indexada nem nas fontes consultadas, então prefiro não responder de memória.",
  tool_error:
    "Não consegui consultar as fontes agora: a busca na documentação ou uma API externa falhou. Tente de novo em instantes.",
  blocked:
    "Não consegui sustentar a resposta nas fontes consultadas, então ela foi retida. Tente reformular a pergunta.",
  blocked_no_lookup:
    "Só respondo com base em fontes consultadas, e esta resposta não usou nenhuma, então foi retida. Pergunte sobre a documentação indexada ou sobre um pacote no npm ou no GitHub.",
  refused: "O modelo recusou este pedido.",
  step_limit:
    "Parei antes de concluir: a pergunta pediu mais consultas do que o limite de passos permite. Tente dividi-la em perguntas menores.",
} as const;

/** Acima disto, resposta sem consulta e sem fonte não passa. */
export const DIRECT_ANSWER_MAX_CHARS = 400;

const CITATION = /\[(\d{1,4})\]/g;

export interface JudgeInput {
  text: string;
  sources: Source[];
  /** Quantas fontes já existiam quando o turno começou. */
  turnSourceStart: number;
  toolCalls: number;
  toolErrors: number;
}

export function judge(input: JudgeInput): Verdict {
  const known = new Map(input.sources.map((source) => [source.n, source]));
  const cited = new Set<number>();
  const text = input.text
    .replace(CITATION, (match, digits: string) => {
      const n = Number(digits);
      if (!known.has(n)) return "";
      cited.add(n);
      return match;
    })
    .replace(/[ \t]+([.,;:!?])/g, "$1")
    .trim();

  if (cited.size > 0) {
    const citations = [...cited]
      .sort((a, b) => a - b)
      .flatMap((n) => known.get(n) ?? []);
    return { outcome: "answered", answer: text, citations };
  }

  if (input.toolCalls > 0) {
    const foundThisTurn = input.sources.length > input.turnSourceStart;
    if (foundThisTurn) return fixed("blocked");
    return fixed(input.toolErrors > 0 ? "tool_error" : "not_found");
  }

  if (text.length > 0 && text.length <= DIRECT_ANSWER_MAX_CHARS) {
    return { outcome: "direct", answer: text, citations: [] };
  }
  return { outcome: "blocked", answer: MESSAGES.blocked_no_lookup, citations: [] };
}

export function fixed(outcome: Exclude<keyof typeof MESSAGES, "blocked_no_lookup">): Verdict {
  return { outcome, answer: MESSAGES[outcome], citations: [] };
}
