/** Uma pergunta pelo terminal: `pnpm ask "como registro um plugin no Fastify?"`. */
import { createRuntime } from "./bootstrap.js";
import { ConfigError, loadConfig, loadEnvFile } from "./config.js";

async function main(): Promise<void> {
  const question = process.argv.slice(2).join(" ").trim();
  if (!question) {
    console.error('uso: pnpm ask "sua pergunta"');
    process.exit(2);
  }
  loadEnvFile();
  const config = loadConfig();
  const log = (details: object, message: string): void => {
    console.error(`[bula] ${message}`, Object.keys(details).length > 0 ? details : "");
  };
  // Na CLI não há por que esperar o serviço subir: ou ele está no ar, ou a pergunta falha logo.
  const { agent } = await createRuntime(config, { info: log, warn: log }, { waitForKnowledgeMs: 0 });

  let streamed = false;
  for await (const event of agent.stream(question)) {
    if (event.type === "step" && event.tools) {
      console.error(`[bula] consultando: ${event.tools.join(", ")}`);
    } else if (event.type === "token") {
      streamed = true;
      process.stderr.write(event.text);
    } else if (event.type === "done") {
      const { result } = event;
      if (streamed) process.stderr.write("\n\n");
      // A resposta final sai no stdout, já passada pelo guardrail; o rascunho foi no stderr.
      console.log(result.answer);
      for (const citation of result.citations) {
        console.log(`[${citation.n}] ${citation.title}\n    ${citation.url}`);
      }
      console.error(
        `[bula] ${result.outcome} · ${result.steps} passo(s) · ` +
          `${result.usage.inputTokens}+${result.usage.outputTokens} tokens · ${result.tookMs} ms`,
      );
    }
  }
}

main().catch((error: unknown) => {
  console.error(error instanceof ConfigError ? error.message : error);
  process.exit(1);
});
