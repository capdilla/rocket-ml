import { resolve, sep } from 'node:path';

const root = resolve(import.meta.dir);
const port = Number(Bun.env.PORT ?? 8765);

const server = Bun.serve({
  hostname: '127.0.0.1',
  port,
  async fetch(request) {
    const url = new URL(request.url);
    let pathname;
    try {
      pathname = decodeURIComponent(url.pathname);
    } catch {
      return new Response('Invalid path', { status: 400 });
    }
    const filePath = resolve(root, `.${pathname === '/' ? '/index.html' : pathname}`);
    if (filePath !== root && !filePath.startsWith(`${root}${sep}`)) {
      return new Response('Invalid path', { status: 403 });
    }
    const file = Bun.file(filePath);
    if (!await file.exists()) return new Response('Not found', { status: 404 });
    return new Response(file, { headers: { 'Content-Type': file.type, 'Cache-Control': 'no-store' } });
  },
});

console.log(`Rocket Lab web: ${server.url}`);
