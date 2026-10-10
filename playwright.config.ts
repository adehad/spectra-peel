import { existsSync } from "node:fs";
import { defineConfig } from "@playwright/test";

// Without these flags Chromium hands WebGPU to SwiftShader, a CPU rasteriser, and the numbers mean nothing.
const REAL_GPU = [
  "--enable-unsafe-webgpu",
  "--ignore-gpu-blocklist",
  "--enable-gpu",
  "--force-high-performance-gpu",
];

// A branded build, because the spike is about what users actually run.
const CHANNEL = process.platform === "win32" ? "msedge" : "chrome";
const CHANNEL_PATHS: Record<typeof CHANNEL, string[]> = {
  chrome: ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/usr/bin/google-chrome"],
  msedge: [
    "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  ],
};
const hasChannel = CHANNEL_PATHS[CHANNEL].some((p) => existsSync(p));

const PORT = Number(process.env["SPECTRA_PORT"] ?? 4173);

export default defineConfig({
  testDir: "e2e",
  timeout: 60_000,
  workers: 1,
  use: {
    baseURL: `http://localhost:${PORT}`,
    browserName: "chromium",
    ...(hasChannel ? { channel: CHANNEL } : {}),
    launchOptions: { args: REAL_GPU },
  },
  webServer: {
    command: `bun run --cwd apps/web build && bun run --cwd apps/web preview --port ${PORT} --strictPort`,
    url: `http://localhost:${PORT}`,
    reuseExistingServer: !process.env["CI"],
    timeout: 120_000,
  },
});
