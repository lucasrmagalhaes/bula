FROM node:22-slim AS build
RUN corepack enable
WORKDIR /app/agent
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY tsconfig.json tsconfig.build.json ./
COPY src ./src
RUN pnpm build && pnpm prune --prod

FROM node:22-slim
ENV NODE_ENV=production AGENT_HOST=0.0.0.0 AGENT_PORT=8200
WORKDIR /app/agent
COPY --from=build --chown=node:node /app/agent/node_modules ./node_modules
COPY --from=build --chown=node:node /app/agent/dist ./dist
COPY --chown=node:node package.json ./
USER node
EXPOSE 8200
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=5 \
  CMD node -e "fetch('http://127.0.0.1:8200/health').then((r) => process.exit(r.ok ? 0 : 1)).catch(() => process.exit(1))"
CMD ["node", "dist/main.js"]
