import { defineConfig } from "vitest/config";

/** Testes que exigem o serviço Python no ar e acesso à rede. Ver test/live/live.test.ts. */
export default defineConfig({
  test: {
    include: ["test/live/**/*.test.ts"],
    environment: "node",
    testTimeout: 60_000,
  },
});
