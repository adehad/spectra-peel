import { cloudflare } from "@cloudflare/vite-plugin";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Multithreaded ORT WASM needs SharedArrayBuffer, which needs cross-origin isolation.
// public/_headers sets the same pair in production.
const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  plugins: [react(), cloudflare()],
  server: { headers: isolation },
  preview: { headers: isolation },
  build: { target: "es2023" },
});
