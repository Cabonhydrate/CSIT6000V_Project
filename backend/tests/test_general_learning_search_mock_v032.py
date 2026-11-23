"""HTTP adapter tests: no live internet required."""
import asyncio
import httpx
from app.agent import background_learning as bg


def test_wikipedia_query_and_no_untrusted_urls(monkeypatch):
    calls=[]
    def handler(request):
        calls.append(str(request.url))
        if 'wikipedia.org' in request.url.host:
            return httpx.Response(200,json={'query':{'pages':[
                {'pageid':123,'index':1,'title':'Q-learning',
                 'extract':'Q-learning is a model-free reinforcement learning algorithm.'}]}})
        raise AssertionError('OpenAlex not expected when Wikipedia provides evidence')
    real=httpx.AsyncClient
    def fake_client(*args,**kwargs):
        kwargs['transport']=httpx.MockTransport(handler)
        return real(*args,**kwargs)
    monkeypatch.setattr(bg.httpx,'AsyncClient',fake_client)
    results=asyncio.run(bg.search_background('请解释 Q-Learning'))
    assert results[0]['source_type']=='external'
    assert results[0]['source_url']=='https://en.wikipedia.org/wiki/Q-learning'
    assert any('gsrsearch=Q-Learning' in c for c in calls)
    assert len(calls)==1


def test_scholarly_fallback_only_uses_verified_doi(monkeypatch):
    def handler(request):
        if 'wikipedia.org' in request.url.host:
            return httpx.Response(200,json={'query':{'pages':[]}})
        return httpx.Response(200,json={'results':[
            {'display_name':'Paper One','doi':'https://doi.org/10.1234/test',
             'abstract_inverted_index':{'We':[0],'study':[1],'learning':[2]}},
            {'display_name':'Bad Link','doi':'http://example.com/unsafe',
             'abstract_inverted_index':{'bad':[0]}}
        ]})
    real=httpx.AsyncClient
    def fake_client(*args,**kwargs):
        kwargs['transport']=httpx.MockTransport(handler)
        return real(*args,**kwargs)
    monkeypatch.setattr(bg.httpx,'AsyncClient',fake_client)
    results=asyncio.run(bg.search_background('请解释强化学习'))
    assert len(results)==1
    assert results[0]['source_url']=='https://doi.org/10.1234/test'
