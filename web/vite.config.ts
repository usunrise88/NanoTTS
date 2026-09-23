import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { tanstackRouter } from '@tanstack/router-plugin/vite'

export default defineConfig({
  // The bundle is embedded in the server binary and can be mounted either at /
  // or behind a reverse proxy at /asr/, so every asset URL has to stay relative.
  base: './',
  plugins: [
    tanstackRouter({ target: 'react', autoCodeSplitting: false }),
    react(),
    tailwindcss(),
  ],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    // One chunk keeps the embedded asset table small and removes a round trip
    // on first paint; the app is far too small for splitting to pay off.
    rollupOptions: { output: { manualChunks: undefined } },
  },
})
