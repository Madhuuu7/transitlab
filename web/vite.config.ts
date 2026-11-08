import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// The console routes in the browser, so there is no API to proxy in
// development. The timetable bundle is served out of public/feed by vite
// exactly as GitHub Pages serves it in production, which means the loading
// path under test locally is the loading path that ships.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist", emptyOutDir: true, sourcemap: false },
  server: { port: 5177 },
  test: {
    // The parity suite decodes a 1.9 MB bundle and runs several hundred
    // one-to-all queries over a 9,948-stop city; the default 5 s is not a
    // meaningful deadline for that.
    testTimeout: 120_000,
    include: ["test/**/*.test.ts"],
  },
});
