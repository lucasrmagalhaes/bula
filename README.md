# Bula

Assistente que responde perguntas sobre documentação técnica e mostra de onde tirou cada afirmação. O nome vem de "ler a bula": ele lê a documentação por você e devolve o trecho, com o link para a versão exata.

São dois serviços:

- **`agent/`** (Node.js, TypeScript, LangGraph.js): o agente. Conversa com o usuário, decide quando buscar na documentação e quando consultar o npm ou o GitHub, e só libera resposta que cite as fontes.
- **`knowledge/`** (Python, FastAPI, LangGraph): o conhecimento. Ingere a documentação, faz a busca híbrida em SQL (Postgres com pgvector) e roda o workflow de RAG, com tracing e avaliação no MLflow.

O índice de exemplo tem a documentação do Fastify v5.12.8 e do Zod v4.6.7.

## Arquitetura

```
 navegador / CLI
       │  HTTP + SSE
       ▼
┌──────────────────────────────── agent (Node) ─────────────────────────────────┐
│  Fastify  ──►  grafo LangGraph.js                                             │
│                                                                               │
│     begin ─► agent ─┬─► tools ─► agent ...     tools:                         │
│                     ├─► finish ─► fim            search_docs ────────────┐    │
│                     └─► limit  ─► fim            npm_package    (npm)    │    │
│                                                  github_latest_release   │    │
│  em código: teto de passos, numeração das fontes, guardrail de saída     │    │
└──────────────────────────────────────────────────────────────────────────┼────┘
                                                                           │ HTTP
┌─────────────────────────────── knowledge (Python) ───────────────────────▼────┐
│  FastAPI  ──►  workflow LangGraph                                             │
│                                                                               │
│     rewrite ─► retrieve ─► grade ─┬─► fim                                     │
│        ▲                          │                                           │
│        └──── nada relevante ──────┘   (até RAG_MAX_ATTEMPTS)                  │
│                                                                               │
│  retrieve = uma consulta SQL: vetorial (pgvector) + textual com IDF, por RRF  │
│  ingestão: GitHub numa tag fixa ─► trechos por seção ─► embeddings ─► Postgres│
│  MLflow: trace de cada execução e runs de avaliação da busca                  │
└───────────────────────────────────────────────────────────────────────────────┘
```

Os dois grafos são de tipos diferentes, de propósito:

- O do **agente** é um agente: o modelo escolhe as tools e quantas vezes consultar.
- O do **RAG** é um workflow: o caminho está escrito no grafo e o modelo só preenche duas decisões tipadas (as consultas de busca e os ids dos trechos relevantes).

## Como rodar

Precisa de Node 22, pnpm, Python 3.12 e [uv](https://docs.astral.sh/uv/). Não precisa de Docker nem de chave de API para o primeiro uso: sem `.env`, o projeto sobe em **modo offline**, com a busca e a API de verdade e sem modelo de linguagem.

```bash
# 1. conhecimento
cd knowledge
uv sync --all-extras              # inclui o Postgres embutido
uv run bula-knowledge ingest      # baixa a documentação e o modelo de embedding (cerca de 220 MB)
uv run bula-knowledge serve       # http://127.0.0.1:8100

# 2. agente, em outro terminal
cd agent
pnpm install
pnpm dev                          # http://127.0.0.1:8200
```

Abra `http://127.0.0.1:8200` para a página de chat, ou pergunte pelo terminal:

```bash
pnpm ask "Como valido um email com o Zod?"
```

Para usar um modelo, copie `.env.example` para `.env` na raiz e preencha `LLM_PROVIDER` e a chave. O mesmo arquivo serve aos dois serviços.

| Provedor | `LLM_PROVIDER` | Modelo |
| --- | --- | --- |
| Anthropic | `anthropic` | padrão `claude-opus-5-5` |
| OpenAI | `openai` | `LLM_MODEL` obrigatório |
| Google | `google` | `LLM_MODEL` obrigatório |
| nenhum | `offline` | o agente devolve os trechos encontrados, sem redigir |

Outros comandos do serviço de conhecimento:

```bash
uv run bula-knowledge search "onRequestAbort" --mode lexical   # busca crua, sem LLM
uv run bula-knowledge context "How do I register a plugin?"    # workflow de RAG
uv run bula-knowledge sources                                  # o que está indexado
uv run bula-knowledge ingest --path ./minhas-docs --id interno # indexa uma pasta local
uv run bula-knowledge eval                                     # avalia a busca e grava no MLflow
uv run bula-knowledge eval --workflow                          # avalia o RAG com LLM (gasta tokens)
uv run mlflow ui --backend-store-uri sqlite:///../data/mlflow.db
```

## Onde está cada coisa

| Assunto | Onde |
| --- | --- |
| Agente em LangGraph.js | [`agent/src/agent/graph.ts`](agent/src/agent/graph.ts), [`agent.ts`](agent/src/agent/agent.ts) |
| Guardrail de saída | [`agent/src/agent/guardrails.ts`](agent/src/agent/guardrails.ts) |
| Integração com LLMs (Anthropic, OpenAI, Google) | [`agent/src/llm/models.ts`](agent/src/llm/models.ts), [`knowledge/src/bula_knowledge/llm.py`](knowledge/src/bula_knowledge/llm.py) |
| Tools sobre APIs externas | [`agent/src/tools/`](agent/src/tools) |
| API com streaming (SSE) | [`agent/src/server/app.ts`](agent/src/server/app.ts) |
| Workflow de RAG em LangGraph (Python) | [`knowledge/src/bula_knowledge/rag/workflow.py`](knowledge/src/bula_knowledge/rag/workflow.py) |
| Busca híbrida em SQL | [`knowledge/src/bula_knowledge/search.py`](knowledge/src/bula_knowledge/search.py) |
| Esquema e pgvector | [`knowledge/src/bula_knowledge/db.py`](knowledge/src/bula_knowledge/db.py) |
| Ingestão e divisão em trechos | [`knowledge/src/bula_knowledge/ingest/`](knowledge/src/bula_knowledge/ingest) |
| API FastAPI | [`knowledge/src/bula_knowledge/api.py`](knowledge/src/bula_knowledge/api.py) |
| MLflow: tracing e avaliação | [`tracking.py`](knowledge/src/bula_knowledge/tracking.py), [`evaluation/retrieval.py`](knowledge/src/bula_knowledge/evaluation/retrieval.py) |
| Contrato entre os serviços | [`schemas.py`](knowledge/src/bula_knowledge/schemas.py) e [`agent/src/knowledge/client.ts`](agent/src/knowledge/client.ts) |

## Decisões

**Busca híbrida, numa consulta SQL.** A busca vetorial erra termo exato (nome de função, opção, código de erro) e a textual erra paráfrase. As duas listas são fundidas por RRF, que soma `1 / (60 + posição)` e dispensa normalizar notas que não se comparam.

**A perna textual tem IDF.** O `ts_rank` do Postgres ignora a raridade do termo: numa pergunta em linguagem natural, o nome da biblioteca casava com quase todo trecho e dominava a lista. Aqui cada termo casado vale o seu IDF, tirado de uma tabela de estatísticas refeita na ingestão, e um trecho só entra se os termos casados cobrirem ao menos 35% do peso da pergunta. Isso cala a perna textual quando a pergunta está em português e o corpus em inglês, em vez de deixá-la devolver ruído.

**Citação conferível.** A ingestão baixa cada fonte do GitHub numa tag fixa e grava o commit. Toda citação é um link para o arquivo naquele commit, com a âncora da seção.

**Quem libera a resposta é o código.** O modelo redige; o guardrail decide. Resposta com citação válida sai, sem as citações que apontam para fonte inexistente. Se houve consulta e nada voltou, sai uma mensagem fixa de "não encontrei" e o texto do modelo é descartado. Se havia fontes e a resposta não cita nenhuma, ela fica retida. Sem consulta, só passa resposta curta.

**Todo laço tem teto fora do modelo.** O agente tem limite de chamadas por pergunta (`AGENT_MAX_STEPS`) e o RAG, de tentativas (`RAG_MAX_ATTEMPTS`). Ao bater o teto do agente, as chamadas de tool pendentes recebem um resultado de "não executada", para o histórico continuar válido no turno seguinte.

**O histórico só recebe acréscimos.** Os modelos atuais da Anthropic amarram os blocos de raciocínio à conversa e recusam histórico editado. Por isso a resposta que o usuário vê fica separada das mensagens, o prompt de sistema não muda durante o processo e o custo de conversa longa é contido por um limite de turnos, não por corte de mensagens antigas.

**Falha de LLM no RAG degrada, não derruba.** Se a reescrita ou a triagem falhar, o passo cai no comportamento sem LLM e a resposta sai marcada com `degraded: true`.

**O índice pertence a um modelo de embedding.** O nome e a dimensão ficam gravados no banco, e o serviço se recusa a subir com um modelo diferente do que construiu o índice.

**Postgres também em desenvolvimento.** Sem `DATABASE_URL`, o serviço sobe um Postgres com pgvector embutido (pacote `pgserver`). O SQL é o mesmo que roda em produção, e os testes rodam contra um Postgres real, sem Docker.

## Avaliação da busca

`bula-knowledge eval` roda um [conjunto ouro](knowledge/eval/golden.jsonl) de 44 perguntas (24 em inglês, 20 em português; 32 em linguagem natural, 12 com um identificador exato) nos três modos e grava cada um como um run no MLflow. Não usa LLM. Medido em 19/11/2026, com 1.181 trechos indexados e o modelo `paraphrase-multilingual-MiniLM-L12-v2` em CPU:

| Modo | Acerto entre os 5 primeiros | MRR |
| --- | --- | --- |
| híbrida | 82% | 0,69 |
| só vetorial | 73% | 0,62 |
| só textual | 45% | 0,37 |

- A busca leva de 12 a 18 ms na mediana e de 21 a 25 ms no percentil 95, nos três modos. A diferença entre eles ficou dentro da variação entre uma execução e outra.
- Em pergunta com identificador exato, a híbrida acerta 12 de 12; a vetorial, 11.
- Em português a híbrida acerta 70%, contra 92% em inglês. O modelo de embedding é pequeno e o corpus é em inglês. A etapa de reescrita do RAG existe para fechar essa distância, e a seção seguinte mede quanto ela fecha.
- A troca para IDF foi medida num índice anterior a este, com as mesmas 44 perguntas: com `ts_rank_cd` puro a híbrida acertava 82% com MRR de 0,62, e com IDF e limiar de cobertura passou a 84% com MRR de 0,73. A tabela acima é de um índice refeito depois, com outro texto vetorizado, então os dois pares de números não se comparam entre si.
- A fórmula da perna textual e o limiar de cobertura foram escolhidos neste mesmo conjunto, então os números são otimistas: não há conjunto separado de validação.

## Avaliação do workflow de RAG

`bula-knowledge eval --workflow` passa as mesmas 44 perguntas pelo workflow com LLM e mede três listas por pergunta: a busca híbrida com a pergunta como veio, os candidatos depois da reescrita e o que sobrou depois da triagem. Medido em 20/11/2026 com `claude-opus-5-5`, esforço `low`, numa execução só:

| Etapa | Acerto entre os 5 primeiros | MRR | Inglês | Português |
| --- | --- | --- | --- | --- |
| busca híbrida, pergunta crua | 82% | 0,69 | 92% | 70% |
| depois da reescrita | 98% | 0,84 | 100% | 95% |
| depois da triagem | 98% | 0,87 | 100% | 95% |

- A reescrita fecha quase toda a distância do português: 19 das 20 perguntas, contra 14 sem ela.
- A triagem não derrubou o acerto e melhorou a posição do trecho certo. Ela aprovou em média 4,75 trechos por pergunta, de até 12 candidatos, e é isso que segue para o agente.
- Custo e latência: 90 chamadas ao modelo, 203 mil tokens de entrada e 3 mil de saída. Mediana de 3,9 s por pergunta e percentil 95 de 4,3 s, contra menos de 20 ms da busca sem LLM. É o preço da reescrita e da triagem.
- Uma pergunta precisou da segunda tentativa, nenhuma ficou sem contexto e nenhuma chamada falhou.
- O único erro (`zod-parse-pt`) pode ser rigor do conjunto ouro: a triagem aprovou dois trechos de outro arquivo do Zod, que não conferi se também respondem.
- Os prompts de reescrita e de triagem foram escritos antes desta medição e não foram ajustados depois dela. Mesmo assim é uma execução só, com 44 perguntas, e modelo de linguagem varia de uma execução para outra.

## Teste com o modelo real

Em 19/11/2026 os dois serviços rodaram com `claude-opus-5-5` (Anthropic) em nove perguntas feitas à mão. Não é uma avaliação: é a conferência de que cada caminho funciona com um modelo de verdade.

| Caso | O que aconteceu | Chamadas ao modelo | Tempo |
| --- | --- | --- | --- |
| Pergunta de documentação, em português | reescrita em três consultas em inglês; resposta com código e quatro citações | 4 | 12 s |
| Segundo turno na mesma conversa | histórico aceito pela API; a numeração das fontes continuou | 4 | 14 s |
| Versão e downloads de um pacote | uma chamada ao npm, resposta citada | 2 | 4 s |
| Pergunta fora do escopo | aviso curto, sem consultar nada | 1 | 2,5 s |
| Função que a documentação não tem | "não encontrei", mais a alternativa que a documentação dá, citada | 4 | 11 s |
| Assunto ausente da documentação | o RAG tentou duas vezes e devolveu "sem contexto"; o agente buscou de novo e respondeu só o que achou | 9 | 20 s |

O workflow de RAG sozinho levou de 4 a 8 segundos por execução, quase tudo nas duas chamadas ao modelo: a busca em si ficou abaixo de 250 ms. No streaming, o primeiro token chegou em 1,9 s. Uma resposta de documentação gastou de 9 a 17 mil tokens de entrada.

O teste também achou um defeito: o guardrail retinha a recusa correta de uma pergunta fora do escopo, porque ela passava de 400 caracteres e não tinha citação. O prompt agora pede esse aviso em uma ou duas frases, e a mensagem de retenção diz o motivo certo.

## Segurança, governança, custo e latência

**Segurança**
- As tools só leem. Nenhuma altera estado fora do processo, o que limita o estrago de uma injeção de prompt vinda de um documento.
- Conteúdo recuperado e resultado de API entram no prompt delimitados e marcados como dado de terceiros, e os dois prompts mandam não seguir instrução que venha dali.
- Argumentos de tool vêm do modelo e são validados antes de virar URL (nome de pacote, `dono/repositório`). Os hosts das tools são fixos.
- A URL da API da Anthropic é sempre explícita: uma `ANTHROPIC_BASE_URL` solta no ambiente não desvia a chave.
- O serviço de conhecimento aceita um token entre serviços (`KNOWLEDGE_API_TOKEN`). A API pública tem limite de taxa, limite de tamanho e validação de entrada, e não devolve detalhe de erro interno.
- Fora do escopo: autenticação de usuário final e autorização por documento. O índice é público por definição.

**Governança**
- Cada fonte tem tag, commit e licença gravados e expostos em `GET /v1/sources`.
- Nenhuma documentação de terceiros é versionada aqui; os testes usam um corpus fictício escrito para eles.
- Cada execução do RAG gera um trace no MLflow (`MLFLOW_TRACING=true`) e devolve o `trace_id`. O `x-request-id` atravessa os dois serviços e aparece nos logs, que são JSON.

**Custo**
- Tetos de passos, de tentativas e de turnos por conversa.
- `LLM_FAST_MODEL` separa o modelo das etapas curtas do RAG do modelo do agente, e `LLM_EFFORT` vem em `low` para elas.
- Cada resposta devolve os tokens gastos, somando os do RAG, e o custo em dólar se os preços estiverem configurados.
- Reingestão só vetoriza documento que mudou (hash do conteúdo).
- Se o cliente desconecta no meio do streaming, o grafo é interrompido.

**Latência**
- Tempo limite em toda chamada externa, com erro tipado.
- As tools pedidas na mesma rodada executam em paralelo.
- O modelo de embedding é carregado na subida, e a vetorização da pergunta tem cache.
- Depois de cada carga a ingestão roda `ANALYZE`: sem isso a busca textual caía num plano ruim até o autovacuum passar (mediana de 90 ms, contra menos de 20 ms depois).

## Testes

```bash
cd knowledge && uv run pytest            # 52 testes, com Postgres embutido
cd agent && pnpm check                   # tipos, lint e 75 testes
cd agent && pnpm test:live               # ponta a ponta, com o serviço Python no ar
```

Nenhum teste precisa de chave. O modelo é trocado por um roteirizado, e os pedidos à Anthropic são conferidos contra um servidor HTTP local que guarda o corpo recebido: é assim que se fixa que não vai `temperature` nem `tool_choice` forçado, que os modelos atuais recusam.

## Limites e o que falta verificar

- **Modelo real**: só a Anthropic foi exercitada, com `claude-opus-5-5`. A recuperação do RAG tem avaliação automática; a resposta final do agente só foi conferida à mão, em nove perguntas. OpenAI e Google têm só a fábrica, nunca chamada.
- **Docker e CI**: o `docker-compose.yml`, os dois `Dockerfile` e o workflow do GitHub Actions foram escritos e ainda não foram executados.
- **Databricks**: o MLflow roda local. Apontar `MLFLOW_TRACKING_URI` para um workspace não foi testado. Registro e serving de modelo não foram feitos.
- **Memória de conversa**: fica em memória do processo, com descarte da menos usada. Com mais de uma réplica, precisa de um checkpointer em Postgres.
- **Modo offline**: é uma demonstração sem LLM. A triagem vira um corte por distância e a pergunta em português perde qualidade.
- **`pgserver`**: serve a desenvolvimento e teste. Produção usa um Postgres de verdade via `DATABASE_URL`.

## Fontes do índice de exemplo

| Fonte | Versão | Licença |
| --- | --- | --- |
| [fastify/fastify](https://github.com/fastify/fastify), `docs/` | v5.12.8 | MIT |
| [colinhacks/zod](https://github.com/colinhacks/zod), `packages/docs/content/` | v4.6.7 | MIT |

Para indexar outra biblioteca, acrescente uma entrada em [`sources.py`](knowledge/src/bula_knowledge/ingest/sources.py) e rode a ingestão.

## Licença

O código é distribuído sob a [licença MIT](LICENSE). Ela não se estende à documentação indexada, que pertence aos respectivos projetos e não faz parte deste repositório.
