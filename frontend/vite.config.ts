import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev-proxy target for the FastAPI backend. Override with VITE_PROXY_TARGET
// when port 8000 is taken (e.g. `VITE_PROXY_TARGET=http://localhost:8001 npm run dev`).
const backend = process.env.VITE_PROXY_TARGET ?? 'http://localhost:8000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/auth': { target: backend, changeOrigin: true },
      '/exercises': { target: backend, changeOrigin: true },
      '/workouts': { target: backend, changeOrigin: true },
      '/programs': { target: backend, changeOrigin: true },
      '/health': { target: backend, changeOrigin: true },
      '/chat': { target: backend, changeOrigin: true },
    },
  },
})
