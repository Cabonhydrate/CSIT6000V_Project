"""Local Ollama native /api/chat + other OpenAI-compatible model serving.

Ollama Qwen3 defaults to thinking. /api/chat with think=false prevents reasoning
from consuming the entire output budget; this fixes an empty-content demo case.
The alternative OpenAI-compatible path remains available for future vLLM.
"""
import asyncio
import json
import time
from urllib.parse import urlparse
import httpx
from ..config import settings
from ..logging_setup import get_logger

logger = get_logger('model')


class ModelUnavailable(RuntimeError):
    pass


class ModelClient:
    def __init__(self, base_url=None, model=None, api_key=None, provider=None):
        self.base_url = (base_url or settings.model_base_url).rstrip('/')
        self.model = model or settings.model_name
        self.api_key = api_key or settings.model_api_key
        selected = provider or settings.model_provider
        parsed = urlparse(self.base_url)
        self.provider = ('ollama' if parsed.port == 11434 else 'openai') if selected == 'auto' else selected
        self.native_base = self.base_url.removesuffix('/v1')
        self.headers = {'Authorization': 'Bearer ' + self.api_key}
        self.last_stream_meta = {}

    async def is_available(self):
        """A green dot means this model is installed and the model server responds."""
        try:
            async with httpx.AsyncClient(timeout=4) as client:
                if self.provider == 'ollama':
                    response = await client.get(self.native_base + '/api/tags')
                    response.raise_for_status()
                    models = response.json().get('models', [])
                    return self.model in {m.get('name') or m.get('model') for m in models}
                response = await client.get(self.base_url + '/models', headers=self.headers)
                response.raise_for_status()
                models = response.json().get('data', [])
                return self.model in {m.get('id') for m in models}
        except (httpx.HTTPError, ValueError, KeyError):
            return False

    def _request(self, messages, temperature, max_tokens, stream, json_mode=False):
        if self.provider == 'ollama':
            payload = {
                'model': self.model, 'messages': messages, 'stream': stream,
                'think': False,
                'options': {'temperature': temperature, 'num_predict': max_tokens},
            }
            if json_mode: payload['format']='json'
            return self.native_base + '/api/chat', payload, {}
        payload = {
            'model': self.model, 'messages': messages, 'temperature': temperature,
            'max_tokens': max_tokens, 'stream': stream,
        }
        if settings.model_reasoning_effort:
            payload['reasoning_effort'] = settings.model_reasoning_effort
        if json_mode: payload['response_format']={'type':'json_object'}
        return self.base_url + '/chat/completions', payload, self.headers

    async def chat(self, messages, temperature=0.0, max_tokens=900, run_id=None, json_mode=False):
        """Non-streamed completion used for Agent evidence assessment."""
        start = time.perf_counter()
        url, payload, headers = self._request(messages, temperature, max_tokens, stream=False, json_mode=json_mode)
        try:
            retry_count = 0
            async with httpx.AsyncClient(timeout=settings.model_timeout) as client:
                for attempt in range(2):
                    response = await client.post(url, headers=headers, json=payload)
                    # A single bounded retry for transient model-server failures.
                    # Persistent 500s (e.g. GPU OOM/model crash) still surface
                    # as errors; no prompt or fact-validation behavior changes.
                    if response.status_code in (429, 500, 502, 503, 504) and attempt == 0:
                        retry_count = 1
                        logger.warning('run=%s model_http_retry status=%d model=%s',
                                       run_id or '-', response.status_code, self.model)
                        await asyncio.sleep(0.35)
                        continue
                    if response.status_code >= 400:
                        print(
                            f"[OLLAMA_HTTP_ERROR] "
                            f"run_id={run_id} "
                            f"json_mode={json_mode} "
                            f"status={response.status_code} "
                            f"body={response.text[:3000]}",
                            flush=True,
                        )

                    response.raise_for_status()
                    data = response.json()
                    break
            if self.provider == 'ollama':
                msg = data.get('message') or {}
                content = msg.get('content') or ''
                usage = {'prompt_tokens': data.get('prompt_eval_count'),
                         'completion_tokens': data.get('eval_count')}
                reason = data.get('done_reason', 'stop')
            else:
                choice = data['choices'][0]
                msg = choice.get('message') or {}
                content = msg.get('content') or ''
                usage = data.get('usage') or {}
                reason = choice.get('finish_reason')
            elapsed = round((time.perf_counter() - start) * 1000, 1)
            logger.info('run=%s chat model=%s provider=%s ms=%s output_chars=%d finish=%s usage=%s',
                        run_id or '-',self.model, self.provider, elapsed, len(content), reason, usage)
            if not content.strip():
                raise ModelUnavailable('The model returned no answer content in non-streaming mode '
                                       f'(finish={reason}). Check model/version and thinking settings.')
            return {'content': content, 'usage': usage, 'latency_ms': elapsed, 'http_retries': retry_count}
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
            logger.exception('run=%s chat error model=%s provider=%s', run_id or '-', self.model, self.provider)
            raise ModelUnavailable(f'Local model request failed: {str(exc)[:240]}') from exc

    async def stream(self, messages, temperature=0.0, max_tokens=1000, run_id=None):
        """Yield *answer* tokens, never reasoning tokens. Record why a stream was empty."""
        start = time.perf_counter()
        url, payload, headers = self._request(messages, temperature, max_tokens, stream=True)
        output_chars, reasoning_chars = 0, 0
        self.last_stream_meta = {}
        finish_reason = None
        usage = {}
        try:
            async with httpx.AsyncClient(timeout=settings.model_timeout) as client:
                async with client.stream('POST', url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        if self.provider == 'ollama':
                            data = json.loads(line)
                            msg = data.get('message') or {}
                            delta = msg.get('content') or ''
                            reasoning_chars += len(msg.get('thinking') or '')
                            if data.get('done'):
                                finish_reason = data.get('done_reason', 'stop')
                                usage = {'prompt_tokens': data.get('prompt_eval_count'),
                                         'completion_tokens': data.get('eval_count')}
                        else:
                            if not line.startswith('data:'):
                                continue
                            body = line[5:].strip()
                            if body == '[DONE]':
                                break
                            data = json.loads(body)
                            choices = data.get('choices') or []
                            if not choices:
                                usage = data.get('usage') or usage
                                continue
                            item = choices[0]
                            msg = item.get('delta') or {}
                            delta = msg.get('content') or ''
                            reasoning_chars += len(msg.get('reasoning') or msg.get('reasoning_content') or '')
                            finish_reason = item.get('finish_reason') or finish_reason
                        if delta:
                            output_chars += len(delta)
                            yield delta
            elapsed = round((time.perf_counter() - start) * 1000, 1)
            self.last_stream_meta = {'finish_reason': finish_reason, 'usage': usage,
                                     'latency_ms': elapsed}
            logger.info('run=%s stream model=%s provider=%s ms=%s output_chars=%d reasoning_chars=%d finish=%s usage=%s',
                        run_id or '-',self.model, self.provider, elapsed, output_chars, reasoning_chars, finish_reason, usage)
            if output_chars == 0:
                raise ModelUnavailable('Model stream returned no answer content '
                                       f'(finish={finish_reason}, reasoning_chars={reasoning_chars}). '
                                       'For Ollama Qwen3 use MODEL_PROVIDER=ollama and /api/chat think=false.')
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
            logger.exception('run=%s stream error model=%s provider=%s', run_id or '-', self.model, self.provider)
            raise ModelUnavailable(f'Local model streaming failed: {str(exc)[:240]}') from exc
