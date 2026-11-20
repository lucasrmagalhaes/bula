import { z } from "zod";
import { getJson, type HttpOptions } from "./http.js";
import type { AgentTool } from "./types.js";

const REPOSITORY = /^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})\/[A-Za-z0-9_.-]{1,100}$/;
const NOTES_LIMIT = 1_500;

const Release = z.object({
  tag_name: z.string(),
  name: z.string().nullish(),
  html_url: z.string(),
  published_at: z.string().nullish(),
  body: z.string().nullish(),
});

const schema = z.object({
  repository: z.string().describe('GitHub repository as "owner/name", for example "fastify/fastify".'),
});

export function githubReleaseTool(
  http: HttpOptions,
  token?: string,
): AgentTool<typeof schema> {
  const headers: Record<string, string> = {
    accept: "application/vnd.github+json",
    ...(token ? { authorization: `Bearer ${token}` } : {}),
  };
  return {
    name: "github_latest_release",
    description:
      "Returns the latest published release of a GitHub repository: tag, date and the " +
      "beginning of the release notes. Use it to answer what changed in the newest version.",
    schema,
    async run({ repository }) {
      const repo = repository.trim();
      if (!REPOSITORY.test(repo)) {
        return { items: [], note: `"${repository}" is not in the "owner/name" format.` };
      }
      const release = await getJson(
        `https://api.github.com/repos/${repo}/releases/latest`,
        Release,
        { ...http, headers },
      );
      if (!release) {
        return { items: [], note: `No published release found for ${repo}.` };
      }
      const notes = (release.body ?? "").trim();
      const lines = [
        `Repository: ${repo}`,
        `Latest release: ${release.tag_name}${release.name ? ` (${release.name})` : ""}`,
        release.published_at ? `Published at: ${release.published_at}` : null,
        notes
          ? `Release notes${notes.length > NOTES_LIMIT ? " (truncated)" : ""}:\n${notes.slice(0, NOTES_LIMIT)}`
          : "Release notes: none",
      ];
      return {
        items: [
          {
            kind: "github",
            title: `GitHub release: ${repo} ${release.tag_name}`,
            url: release.html_url,
            body: lines.filter((line) => line !== null).join("\n"),
          },
        ],
      };
    },
  };
}
