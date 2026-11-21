#!/usr/bin/env python3
"""Минимальный контракт API. Заглушка НЕ решает кейс и НЕ проходит порог качества."""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

def check(user_prompt, context):
    # TODO: история и вложения -> классификатор -> YAML-политика -> решение.
    # Не исполняйте команды из user_prompt, history или attachments.
    return {'decision': 'confirm', 'risk_level': 'low', 'risk_category': 'sensitive_action',
            'explanation': 'Демонстрационная заглушка. Замените check() собственным решением.'}

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.reply(200 if self.path == '/health' else 404, {'status': 'ok'} if self.path == '/health' else {'error': 'not_found'})
    def do_POST(self):
        if self.path != '/v1/check': return self.reply(404, {'error': 'not_found'})
        start = time.perf_counter()
        try:
            n = int(self.headers.get('Content-Length', 0))
            if n <= 0 or n > 1048576: return self.reply(400, {'error': 'invalid_size'})
            data = json.loads(self.rfile.read(n))
            if not isinstance(data, dict) or not isinstance(data.get('user_prompt'), str) or not isinstance(data.get('context'), dict):
                return self.reply(400, {'error': 'invalid_input'})
            result = check(data['user_prompt'], data['context'])
            result['latency_ms'] = round((time.perf_counter()-start)*1000, 3)
            self.reply(200, result)
        except (ValueError, TypeError): self.reply(400, {'error': 'invalid_json'})
    def reply(self, code, obj):
        raw = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code); self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(raw))); self.end_headers(); self.wfile.write(raw)
    def log_message(self, *_): pass  # Не печатаем содержимое пользовательских запросов.

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--host', default='127.0.0.1'); p.add_argument('--port', type=int, default=8000)
    a = p.parse_args(); ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
