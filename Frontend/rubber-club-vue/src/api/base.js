export function getApiBase() {
    // 优先读取 Vite 环境变量
    const env = import.meta?.env?.VITE_API_BASE || '';
    // 回退到本地默认端口（本地开发时生效）
    const fallback = 'http://127.0.0.1:5001';
    // 选取非空的那个，并去掉结尾斜杠
    const base = (env || fallback).replace(/\/$/, '')
    console.log('[build mode]', import.meta.env.MODE, 'VITE_API_BASE =', import.meta.env.VITE_API_BASE);
    return base;
}