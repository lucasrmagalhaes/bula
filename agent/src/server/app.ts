/**
 * API HTTP do agente.
 *
 *   GET  /                 página de chat (um arquivo estático)
 *   GET  /health           estado do serviço
 *   POST /v1/chat          resposta inteira em JSON
 *   POST /v1/chat/stream   a mesma resposta em Server-Sent Events
 */
import { randomUUID } from "node:crypto";
import { readFile } from "node:fs/promises";
import rateLimit from "@fastify/rate-limit";
import Fastify, { type FastifyInstance, type FastifyReply } from "fastify";
import { z } from "zod";
import { ThreadFullError, type Agent, type AgentEvent } from "../agent/agent.js";
import type { Config } from "../config.js";

const PAGE = new URL("./chat.html", import.meta.url);

export interface AppDeps {
  agent: Agent;
  config: Config;
  indexed: string[];
}

export async function buildApp(deps: AppDeps): Promise<FastifyInstance> {
  const { agent, config } = deps;

  const ChatBody = z.object({
    message: z.string().trim().min(1).max(config.agent.maxMessageChars),
    threadId: z.uuid().optional(),
  });

  const app = Fastify({
    logger: {
      level: config.server.logLevel,
      redact: ["req.headers.authorization", "req.headers.cookie"],
    },
    bodyLimit: 64 * 1024,
    genReqId: (request) => {
      const header = request.headers["x-request-id"];
      return typeof header === "string" && header.length <= 64 ? header : randomUUID();
    },
  });

  await app.register(rateLimit, {
    global: false,
    max: config.server.rateLimitPerMinute,
    timeWindow: "1 minute",
  });
  const limited = { config: { rateLimit: {} } };

  app.addHook("onSend", (request, reply, _payload, done) => {
    reply.header("x-request-id", request.id);
    reply.header("x-content-type-options", "nosniff");
    done();
  });

  app.get("/health", () => ({
    status: "ok",
    llm: { provider: config.llm.provider, model: config.llm.model || null },
    indexed: deps.indexed,
  }));

  app.get("/", async (_request, reply) => {
    return reply
      .type("text/html; charset=utf-8")
      .header(
        "content-security-policy",
        "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; form-action 'none'",
      )
      .send(await readFile(PAGE, "utf8"));
  });

  app.post("/v1/chat", limited, async (request, reply) => {
    const body = ChatBody.safeParse(request.body);
    if (!body.success) return badRequest(reply, body.error);
    try {
      const result = await agent.ask(body.data.message, {
        threadId: body.data.threadId,
        requestId: request.id,
      });
      request.log.info(summary(result), "chat");
      return result;
    } catch (error) {
      if (error instanceof ThreadFullError) {
        return reply.code(409).send({ error: "thread_full", message: error.message });
      }
      throw error;
    }
  });

  app.post("/v1/chat/stream", limited, async (request, reply) => {
    const body = ChatBody.safeParse(request.body);
    if (!body.success) return badRequest(reply, body.error);

    // Se o cliente desconectar, o grafo para em vez de seguir gastando tokens. O evento
    // é o da resposta: o `close` do pedido dispara assim que o corpo termina de ser lido.
    const abort = new AbortController();
    reply.raw.on("close", () => {
      if (!reply.raw.writableFinished) abort.abort();
    });

    reply.hijack();
    reply.raw.writeHead(200, {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache, no-transform",
      connection: "keep-alive",
      "x-accel-buffering": "no",
      "x-request-id": request.id,
    });
    const send = (event: string, data: unknown): void => {
      reply.raw.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
    };

    try {
      const events = agent.stream(body.data.message, {
        threadId: body.data.threadId,
        requestId: request.id,
        signal: abort.signal,
      });
      for await (const event of events) {
        send(event.type, payload(event));
        if (event.type === "done") request.log.info(summary(event.result), "chat");
      }
    } catch (error) {
      if (abort.signal.aborted) {
        request.log.info("cliente desconectou; turno interrompido");
      } else if (error instanceof ThreadFullError) {
        send("error", { error: "thread_full", message: error.message });
      } else {
        request.log.error({ err: error }, "falha no turno");
        send("error", { error: "internal", message: "falha ao responder" });
      }
    } finally {
      reply.raw.end();
    }
  });

  app.setErrorHandler((error: Error & { statusCode?: number }, request, reply) => {
    const status = error.statusCode ?? 500;
    if (status >= 500) request.log.error({ err: error }, "erro não tratado");
    // Erro interno não vaza detalhe para o cliente; o id permite achar o log.
    return reply.code(status).send({
      error: status === 429 ? "rate_limited" : status >= 500 ? "internal" : "bad_request",
      message: status >= 500 ? "erro interno" : error.message,
      requestId: request.id,
    });
  });

  return app;
}

function badRequest(reply: FastifyReply, error: z.ZodError): FastifyReply {
  return reply.code(400).send({
    error: "bad_request",
    message: error.issues.map((issue) => `${issue.path.join(".")}: ${issue.message}`).join("; "),
  });
}

function payload(event: AgentEvent): unknown {
  if (event.type === "done") return event.result;
  if (event.type === "token") return { text: event.text };
  return { node: event.node, tools: event.tools };
}

function summary(result: Extract<AgentEvent, { type: "done" }>["result"]): object {
  return {
    threadId: result.threadId,
    outcome: result.outcome,
    steps: result.steps,
    citations: result.citations.length,
    usage: result.usage,
    tookMs: result.tookMs,
  };
}
