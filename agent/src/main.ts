import { createRuntime } from "./bootstrap.js";
import { ConfigError, loadConfig, loadEnvFile } from "./config.js";
import { buildApp } from "./server/app.js";

async function main(): Promise<void> {
  loadEnvFile();
  const config = loadConfig();

  // O logger do Fastify só existe depois do app; a montagem usa um provisório no mesmo formato.
  const pending: { level: "info" | "warn"; details: object; message: string }[] = [];
  const runtime = await createRuntime(config, {
    info: (details, message) => pending.push({ level: "info", details, message }),
    warn: (details, message) => pending.push({ level: "warn", details, message }),
  });

  const app = await buildApp({ agent: runtime.agent, config, indexed: runtime.indexed });
  for (const entry of pending) app.log[entry.level](entry.details, entry.message);

  for (const signal of ["SIGINT", "SIGTERM"] as const) {
    process.once(signal, () => {
      app.log.info({ signal }, "encerrando");
      void app.close().then(() => process.exit(0));
    });
  }

  await app.listen({ host: config.server.host, port: config.server.port });
}

main().catch((error: unknown) => {
  if (error instanceof ConfigError) {
    console.error(error.message);
  } else {
    console.error(error);
  }
  process.exit(1);
});
