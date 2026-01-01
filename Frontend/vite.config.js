import { fileURLToPath, URL } from 'node:url'
import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig(({ mode }) => {
  // 通过 loadEnv 读取 .env 文件（含 .env.local）
  const env = loadEnv(mode, process.cwd(), '')
  const BASE = env.VITE_BASE || '/'
  
  return {
  base: BASE,
  plugins: [vue()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    port: 5173,
    hmr: { overlay: false },
    allowedHosts: ['app.rubberclub.app'],
  },
}
})