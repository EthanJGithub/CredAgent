import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, proxy /api to the local FastAPI backend so the app can use relative
// URLs. In production, set VITE_API_BASE to the deployed API origin instead.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Use 127.0.0.1 (not localhost) so it doesn't resolve to IPv6 ::1 on
      // Windows while uvicorn is bound to IPv4 127.0.0.1.
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
