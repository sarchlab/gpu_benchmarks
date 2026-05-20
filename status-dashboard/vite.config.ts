import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    allowedHosts: ["yss2.tail13b378.ts.net"],
    proxy: {
      "/api": "http://127.0.0.1:4174"
    }
  }
});
