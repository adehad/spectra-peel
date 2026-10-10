import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    projects: [
      {
        test: {
          name: "engine",
          root: "packages/engine",
          environment: "node",
          include: ["src/**/*.test.ts"],
        },
      },
    ],
  },
});
