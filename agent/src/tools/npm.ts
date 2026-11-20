import { z } from "zod";
import { getJson, type HttpOptions } from "./http.js";
import { ToolError, type AgentTool } from "./types.js";

// Regra de nome do npm. Valida antes de montar a URL: o nome vem do modelo.
const PACKAGE_NAME = /^(?:@[a-z0-9-~][a-z0-9-._~]*\/)?[a-z0-9-~][a-z0-9-._~]*$/;

const Latest = z.object({
  name: z.string(),
  version: z.string(),
  description: z.string().optional(),
  license: z.union([z.string(), z.object({ type: z.string() })]).optional(),
  homepage: z.string().optional(),
  engines: z.record(z.string(), z.string()).optional(),
});

const Downloads = z.object({ downloads: z.number(), start: z.string(), end: z.string() });

const schema = z.object({
  name: z
    .string()
    .describe('Exact npm package name, for example "fastify" or "@langchain/core".'),
});

export function npmPackageTool(http: HttpOptions): AgentTool<typeof schema> {
  return {
    name: "npm_package",
    description:
      "Looks up a package on the npm registry: latest published version, license, " +
      "description and downloads in the last week. Use it for current facts about a " +
      "package; the documentation index does not carry current versions.",
    schema,
    async run({ name }) {
      const normalized = name.trim().toLowerCase();
      if (!PACKAGE_NAME.test(normalized) || normalized.length > 214) {
        return { items: [], note: `"${name}" is not a valid npm package name.` };
      }
      const path = normalized.replace("/", "%2F");
      const latest = await getJson(`https://registry.npmjs.org/${path}/latest`, Latest, http);
      if (!latest) return { items: [], note: `No package named "${normalized}" on npm.` };

      // Os downloads são um complemento: se essa chamada falhar, a versão ainda responde.
      const downloads = await getJson(
        `https://api.npmjs.org/downloads/point/last-week/${normalized}`,
        Downloads,
        http,
      ).catch((error: unknown) => {
        if (error instanceof ToolError) return null;
        throw error;
      });

      const license = typeof latest.license === "string" ? latest.license : latest.license?.type;
      const lines = [
        `Package: ${latest.name}`,
        `Latest version: ${latest.version}`,
        license ? `License: ${license}` : null,
        latest.description ? `Description: ${latest.description}` : null,
        latest.engines?.node ? `Node.js required: ${latest.engines.node}` : null,
        downloads
          ? `Downloads from ${downloads.start} to ${downloads.end}: ${downloads.downloads}`
          : "Downloads: unavailable right now",
      ];
      return {
        items: [
          {
            kind: "npm",
            title: `npm: ${latest.name}`,
            url: `https://www.npmjs.com/package/${latest.name}`,
            body: lines.filter((line) => line !== null).join("\n"),
          },
        ],
      };
    },
  };
}
