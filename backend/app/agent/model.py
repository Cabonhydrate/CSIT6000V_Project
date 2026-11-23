"""Local Ollama /api/chat and OpenAI-compatible model serving.

Ollama Qwen3 is called with think=False, as in the original implementation.
When Ollama aborts generation because of repeated tokens, retry exactly once
with a small, controlled decoding adjustment instead of replaying the identical
request. JSON mode, prompts and output limits are preserved.
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
    """A model request could not produce a usable response."""


class ModelOutputRepetition(ModelUnavailable):
    """The inference server aborted generation after detecting repeated tokens."""


class ModelClient:
    _RETRY_STATUSES = (429, 500, 502, 503, 504)
    _REPEAT_ERROR_TEXT = 'token repeat limit reached'

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
        """Report availability only when the configured model is actually installed."""
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
                'model': self.model,
                'messages': messages,
                'stream': stream,
                'think': False,
                'options': {
                    'temperature': temperature,
                    'num_predict': max_tokens,
                },
            }
            if json_mode:
                payload['format'] = 'json'
            return self.native_base + '/api/chat', payload, {}

        payload = {
            'model': self.model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': stream,
        }
        if settings.model_reasoning_effort:
            payload['reasoning_effort'] = settings.model_reasoning_effort
        if json_mode:
            payload['response_format'] = {'type': 'json_object'}
        return self.base_url + '/chat/completions', payload, self.headers

    @staticmethod
    def _error_message(response):
        """Keep the server's actual error detail; HTTP status text is not enough."""
        try:
            data = response.json()
            if isinstance(data, dict):
                error = data.get('error')
                if isinstance(error, dict):
                    error = error.get('message') or str(error)
                if error:
                    return str(error)[:500]
        except (ValueError, TypeError):
            pass
        return response.text.strip()[:500] or response.reason_phrase

    @classmethod
    def _is_repeat_error(cls, message):
        return cls._REPEAT_ERROR_TEXT in message.lower()

    def _http_failure(self, response, run_id, json_mode, stage):
        detail = self._error_message(response)
        logger.error(
            'run=%s model_http_error stage=%s provider=%s model=%s '
            'status=%d json_mode=%s error=%r',
            run_id or '-', stage, self.provider, self.model,
            response.status_code, json_mode, detail,
        )
        if self._is_repeat_error(detail):
            return ModelOutputRepetition(
                f'Model output repetition (HTTP {response.status_code}): {detail}'
            )
        return ModelUnavailable(
            f'Model server HTTP {response.status_code}: {detail}'
        )

    def _repeat_retry_payload(self, payload, temperature):
        """A one-off Ollama retry to break deterministic repetition.

        Do NOT remove format='json', change the prompt or increase max_tokens.
        These settings apply only after a confirmed token-repeat HTTP error.
        """
        retry_payload = dict(payload)
        options = dict(payload.get('options') or {})
        options['temperature'] = max(float(temperature), 0.3)
        options['repeat_penalty'] = 1.12
        options['repeat_last_n'] = 128
        retry_payload['options'] = options
        return retry_payload

    async def chat(self, messages, temperature=0.0, max_tokens=900, run_id=None, json_mode=False):
        """Non-streamed completion; preserve the existing return contract."""
        start = time.perf_counter()
        url, payload, headers = self._request(
            messages, temperature, max_tokens, stream=False, json_mode=json_mode
        )
        retry_count = 0
        current_payload = payload

        try:
            async with httpx.AsyncClient(timeout=settings.model_timeout) as client:
                for attempt in range(2):
                    response = await client.post(url, headers=headers, json=current_payload)

                    if response.status_code >= 400:
                        detail = self._error_message(response)
                        if attempt == 0 and response.status_code in self._RETRY_STATUSES:
                            retry_count = 1
                            repeat = self.provider == 'ollama' and self._is_repeat_error(detail)
                            if repeat:
                                current_payload = self._repeat_retry_payload(payload, temperature)
                            logger.warning(
                                'run=%s model_http_retry status=%d model=%s '
                                'json_mode=%s repeat_detected=%s strategy=%s error=%r',
                                run_id or '-', response.status_code, self.model,
                                json_mode, repeat,
                                'adjust_decoding' if repeat else 'same_request', detail,
                            )
                            await asyncio.sleep(0.35)
                            continue
                        raise self._http_failure(response, run_id, json_mode, stage='chat')

                    data = response.json()
                    if not isinstance(data, dict):
                        raise ModelUnavailable('Model server returned a non-object JSON response')
                    if data.get('error'):
                        detail = str(data['error'])
                        if self._is_repeat_error(detail):
                            raise ModelOutputRepetition(f'Model output repetition: {detail[:500]}')
                        raise ModelUnavailable(f'Model server error: {detail[:500]}')
                    break

            if self.provider == 'ollama':
                msg = data.get('message') or {}
                content = msg.get('content') or ''
                usage = {
                    'prompt_tokens': data.get('prompt_eval_count'),
                    'completion_tokens': data.get('eval_count'),
                }
                reason = data.get('done_reason', 'stop')
            else:
                choice = data['choices'][0]
                msg = choice.get('message') or {}
                content = msg.get('content') or ''
                usage = data.get('usage') or {}
                reason = choice.get('finish_reason')

            elapsed = round((time.perf_counter() - start) * 1000, 1)
            if not isinstance(content, str):
                raise ModelUnavailable('Model returned non-text answer content')
            logger.info(
                'run=%s chat model=%s provider=%s ms=%s output_chars=%d '
                'finish=%s usage=%s http_retries=%d json_mode=%s',
                run_id or '-', self.model, self.provider, elapsed, len(content),
                reason, usage, retry_count, json_mode,
            )
            if not content.strip():
                raise ModelUnavailable(
                    'The model returned no answer content in non-streaming mode '
                    f'(finish={reason}). Check model/version and thinking settings.'
                )
            return {
                'content': content,
                'usage': usage,
                'latency_ms': elapsed,
                'http_retries': retry_count,
            }

        except ModelUnavailable:
            raise
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            logger.exception(
                'run=%s chat error model=%s provider=%s',
                run_id or '-', self.model, self.provider,
            )
            raise ModelUnavailable(f'Local model request failed: {str(exc)[:240]}') from exc

    async def stream(self, messages, temperature=0.0, max_tokens=1000, run_id=None):
        """Yield answer tokens only; do not replay a partially delivered stream."""
        start = time.perf_counter()
        url, payload, headers = self._request(messages, temperature, max_tokens, stream=True)
        output_chars, reasoning_chars = 0, 0
        self.last_stream_meta = {}
        finish_reason = None
        usage = {}
        ollama_done = False

        try:
            async with httpx.AsyncClient(timeout=settings.model_timeout) as client:
                async with client.stream('POST', url, headers=headers, json=payload) as response:
                    if response.status_code >= 400:
                        await response.aread()
                        raise self._http_failure(response, run_id, False, stage='stream')
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        if self.provider == 'ollama':
                            data = json.loads(line)
                            if 'error' in data:
                                detail = str(data['error'])
                                logger.error(
                                    'run=%s model_stream_error model=%s error=%r',
                                    run_id or '-', self.model, detail[:500],
                                )
                                if self._is_repeat_error(detail):
                                    raise ModelOutputRepetition(
                                        f'Model output repetition during streaming: {detail[:500]}'
                                    )
                                raise ModelUnavailable(f'Model streaming error: {detail[:500]}')
                            msg = data.get('message') or {}
                            delta = msg.get('content') or ''
                            reasoning_chars += len(msg.get('thinking') or '')
                            if data.get('done'):
                                ollama_done = True
                                finish_reason = data.get('done_reason', 'stop')
                                usage = {
                                    'prompt_tokens': data.get('prompt_eval_count'),
                                    'completion_tokens': data.get('eval_count'),
                                }
                        else:
                            if not line.startswith('data:'):
                                continue
                            body = line[5:].strip()
                            if body == '[DONE]':
                                break
                            data = json.loads(body)
                            if data.get('error'):
                                raise ModelUnavailable(
                                    f'Model streaming error: {str(data["error"])[:500]}'
                                )
                            choices = data.get('choices') or []
                            if not choices:
                                usage = data.get('usage') or usage
                                continue
                            item = choices[0]
                            msg = item.get('delta') or {}
                            delta = msg.get('content') or ''
                            reasoning_chars += len(
                                msg.get('reasoning') or msg.get('reasoning_content') or ''
                            )
                            finish_reason = item.get('finish_reason') or finish_reason
                        if delta:
                            output_chars += len(delta)
                            yield delta

            elapsed = round((time.perf_counter() - start) * 1000, 1)
            self.last_stream_meta = {
                'finish_reason': finish_reason,
                'usage': usage,
                'latency_ms': elapsed,
            }
            logger.info(
                'run=%s stream model=%s provider=%s ms=%s '
                'output_chars=%d reasoning_chars=%d finish=%s usage=%s',
                run_id or '-', self.model, self.provider, elapsed,
                output_chars, reasoning_chars, finish_reason, usage,
            )
            if self.provider == 'ollama' and not ollama_done:
                raise ModelUnavailable(
                    'Ollama stream ended without a done=true frame; answer may be incomplete'
                )
            if output_chars == 0:
                raise ModelUnavailable(
                    'Model stream returned no answer content '
                    f'(finish={finish_reason}, reasoning_chars={reasoning_chars}). '
                    'For Ollama Qwen3 use MODEL_PROVIDER=ollama and /api/chat think=false.'
                )

        except ModelUnavailable:
            raise
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            logger.exception(
                'run=%s stream error model=%s provider=%s',
                run_id or '-', self.model, self.provider,
            )
            raise ModelUnavailable(f'Local model streaming failed: {str(exc)[:240]}') from exc
