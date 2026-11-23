"""General regression cases; no external benchmark questions or answer keys."""
import asyncio
import httpx
import pytest

from app.agent.model import ModelClient, ModelUnavailable
from app.skills.math_support import _literal_equation, safe_math_fallback


def _paper():
    # Generic synthetic academic term, with PDF-extraction damage deliberately
    # mimicking a multiline indexed-union fraction.
    return [{'content': (
        'Metric Coverage (MC) evaluates how well a query suite is exercised. '
        'MC(𝑇) =\n\x0c\x0cÐ\n𝑞∈𝑇𝑅𝑘(𝑞)\n\x0c\x0c\n|C|\n'
        'where T is a test suite and C contains all chunks.'
    )}]


def test_math_recovery_supports_full_term_name_without_acronym():
    question = '判断 Metric Coverage 的公式是否正确，请写出论文中的正式定义。'
    result = _literal_equation(question, _paper())
    assert result == (1, 'MC(T) = |∪_{q∈T} R_k(q)| / |C|')
    assert 'MC(T)' in safe_math_fallback(question, _paper()).text


def test_math_recovery_does_not_guess_undefined_acronym():
    question = '判断 Different Metric 的公式是否正确，请写出论文中的正式定义。'
    assert _literal_equation(question, _paper()) is None
    assert safe_math_fallback(question, _paper()) is None


def _client_with_handler(monkeypatch, handler):
    real = httpx.AsyncClient
    monkeypatch.setattr('app.agent.model.httpx.AsyncClient',
                        lambda **kwargs: real(transport=httpx.MockTransport(handler), **kwargs))
    return ModelClient(base_url='http://127.0.0.1:11434/v1', model='qwen3:4b')


def test_single_http_500_can_recover(monkeypatch):
    calls = []
    def handler(req):
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(500, text='initial failure')
        return httpx.Response(200, json={'message': {'content': 'Answer'}, 'done_reason': 'stop'})
    client = _client_with_handler(monkeypatch, handler)
    response = asyncio.run(client.chat([{'role':'user','content':'Hi'}]))
    assert response['content'] == 'Answer'
    assert response['http_retries'] == 1
    assert len(calls) == 2


def test_persistent_http_500_is_not_hidden(monkeypatch):
    calls=[]
    def handler(req):
        calls.append(req)
        return httpx.Response(500,text='still unavailable')
    client = _client_with_handler(monkeypatch,handler)
    with pytest.raises(ModelUnavailable, match='500 Internal Server Error'):
        asyncio.run(client.chat([{'role':'user','content':'Hi'}]))
    assert len(calls) == 2


def test_http_400_not_retried(monkeypatch):
    calls=[]
    def handler(req):
        calls.append(req)
        return httpx.Response(400,text='invalid model options')
    client = _client_with_handler(monkeypatch,handler)
    with pytest.raises(ModelUnavailable,match='400 Bad Request'):
        asyncio.run(client.chat([{'role':'user','content':'Hi'}]))
    assert len(calls) == 1
