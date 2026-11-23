"""Run from backend/: python -m app.diagnose
Checks actual local model responses, without needing any paper or frontend.
"""
import asyncio
from .agent.model import ModelClient, ModelUnavailable


async def main():
    model=ModelClient()
    print(f'Model: {model.model} | provider: {model.provider} | endpoint: {model.base_url}')
    available=await model.is_available()
    print('Installed and server responding:', available)
    if not available:
        print('Check ollama list / ollama pull, then MODEL_NAME and MODEL_BASE_URL.')
        return
    messages=[{'role':'user','content':'What does RAG stand for? Answer in one sentence.'}]
    try:
        result=await model.chat(messages,max_tokens=120)
        print('Non-streaming answer:',result['content'][:400])
        chunks=[]
        async for chunk in model.stream(messages,max_tokens=160):
            chunks.append(chunk)
        print('Streaming answer:', ''.join(chunks)[:400])
        print('PASS: both model request types returned an answer.')
    except ModelUnavailable as exc:
        print('FAILED:',exc)
        print('Check data/logs/app.log for detailed model diagnostics.')


if __name__=='__main__':
    asyncio.run(main())
