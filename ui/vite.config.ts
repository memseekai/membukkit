import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Build directly into the Python package so `membukkit ui` serves the bundle
// with no Node required at runtime.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../src/membukkit/ui_dist",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8377",
    },
  },
});
