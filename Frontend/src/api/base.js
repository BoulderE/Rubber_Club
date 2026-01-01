export function getApiBase() {
    const env = import.meta?.env?.VITE_API_BASE_URL || '';
    const fallback = 'http://127.0.0.1:5001';
    const base = (env || fallback).replace(/\/$/, '')
    console.log('[build mode]', import.meta.env.MODE, 'VITE_API_BASE_URL =', import.meta.env.VITE_API_BASE_URL);
    return base;
}