import { defineConfig } from "cf/config";

export default defineConfig({
  worker: {
    name: "spectra-peel",
    compatibilityDate: "2026-09-17",
    assets: { notFoundHandling: "single-page-application" },
  },
});
