import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["test/**/*.test.ts"],
    exclude: ["test/live/**"],
    environment: "node",
    // O primeiro teste de cada arquivo paga a carga do LangChain e do SDK do provedor.
    // Com a máquina ocupada isso já passou dos 5 s padrão e derrubou teste que estava certo.
    testTimeout: 30_000,
    // Os testes não leem o .env da máquina: cada um monta a própria configuração.
    env: { BULA_SKIP_ENV_FILE: "1" },
  },
});
