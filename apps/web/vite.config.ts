import { resolve } from "node:path";
import { cloudflare } from "@cloudflare/vite-plugin";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { VitePWA } from "vite-plugin-pwa";

// Multithreaded ORT WASM needs SharedArrayBuffer, which needs cross-origin isolation.
// public/_headers sets the same pair in production.
const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

/**
 * `cf build` (it sets this variable) writes the client to the Build Output Specification's assets
 * folder, but vite-plugin-pwa writes the service worker and globs its precache list in the top-level
 * build.outDir. Pointing that at the same folder puts sw.js in the deploy.
 */
const outDir = process.env["CLOUDFLARE_VITE_FORCE_BUILD_OUTPUT"]
  ? ".cloudflare/output/v0/workers/default/assets"
  : "dist";

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    cloudflare(),
    VitePWA({
      // "prompt" never activates a new service worker by itself; the app asks, and not while a job runs.
      registerType: "prompt",
      injectRegister: false,
      manifest: {
        name: "Spectra Peel",
        short_name: "Spectra Peel",
        description: "On-device audio noise removal and source separation",
        theme_color: "#161821",
        background_color: "#161821",
        display: "standalone",
        start_url: "/",
        icons: [
          { src: "/icons/icon-192.png", sizes: "192x192", type: "image/png" },
          { src: "/icons/icon-512.png", sizes: "512x512", type: "image/png" },
          { src: "/icons/maskable-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
        ],
      },
      workbox: {
        globPatterns: ["**/*.{js,css,html,wasm,onnx,png,svg,webmanifest}"],
        // The ORT wasm build is 17 MB and the denoise model 9 MB; both must be cached to work offline.
        maximumFileSizeToCacheInBytes: 32 * 1024 * 1024,
        navigateFallback: "/index.html",
      },
    }),
  ],
  server: { headers: isolation },
  preview: { headers: isolation },
  build: {
    outDir,
    target: "es2023",
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, "index.html"),
      },
    },
  },
  worker: { format: "es" },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
