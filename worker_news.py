import asyncio
import json
import feedparser
import time
import aiohttp
from loguru import logger
from rabbitmq_client import RabbitMQClient, RABBITMQ_URL
from datetime import datetime, timedelta

# Кэш для последних новостей
_news_cache = {
    'entries': [],
    'timestamp': 0
}

async def fetch_rss_news(feed_url="https://www.coindesk.com/arc/outboundfeeds/rss/", max_entries=5):
    """Получает последние новости из RSS ленты."""
    loop = asyncio.get_event_loop()
    feed = await loop.run_in_executor(None, lambda: feedparser.parse(feed_url))
    entries = []
    for entry in feed.entries[:max_entries]:
        entries.append({
            'title': entry.title,
            'summary': entry.summary,
            'published': entry.get('published', ''),
            'link': entry.link
        })
    return entries

async def get_recent_news(hours=6, max_entries=5):
    """Возвращает новости, опубликованные за последние hours часов, с кэшированием."""
    global _news_cache
    now = time.time()
    # Обновляем кэш раз в час
    if not _news_cache['entries'] or (now - _news_cache['timestamp']) > 3600:
        _news_cache['entries'] = await fetch_rss_news(max_entries=10)
        _news_cache['timestamp'] = now
    
    # Фильтруем по времени
    cutoff = datetime.now() - timedelta(hours=hours)
    recent = []
    for e in _news_cache['entries']:
        try:
            pub_time = datetime.strptime(e['published'], '%a, %d %b %Y %H:%M:%S %z')
            pub_time = pub_time.replace(tzinfo=None)
            if pub_time > cutoff:
                recent.append(e)
        except:
            # Если не можем распарсить время, включаем все
            recent.append(e)
    return recent[:max_entries]

async def query_vllm(prompt: str, log, max_tokens=256, retries=3) -> dict:
    url = "http://localhost:8000/v1/completions"
    payload = {
        "model": "/mnt/ssd_combined/models/llama-3.3-70b-awq",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "top_p": 0.9
    }
    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=90) as resp:  # таймаут 60 секунд
                    if resp.status != 200:
                        log.error(f"vLLM error {resp.status}, attempt {attempt+1}/{retries}")
                        if attempt < retries - 1:
                            await asyncio.sleep(2 ** attempt)  # exponential backoff
                            continue
                        return None
                    result = await resp.json()
                    text = result['choices'][0]['text'].strip()
                    import re
                    # Ищем JSON в ответе
                    json_blocks = re.findall(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
                    if json_blocks:
                        try:
                            return json.loads(json_blocks[-1])
                        except:
                            pass
                    json_match = re.search(r'(\{.*?\})', text, re.DOTALL)
                    if json_match:
                        try:
                            return json.loads(json_match.group(1))
                        except:
                            pass
                    log.error(f"Failed to parse JSON: {text}")
                    return None
        except asyncio.TimeoutError:
            log.warning(f"vLLM timeout, attempt {attempt+1}/{retries}")
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                return None
        except Exception as e:
            log.error(f"vLLM request failed: {e}, attempt {attempt+1}/{retries}")
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                return None
    return None

async def process_task(task: dict) -> dict:
    correlation_id = task.get('correlation_id', 'unknown')
    log = logger.bind(correlation_id=correlation_id)

    news = await get_recent_news(hours=6, max_entries=5)
    if not news:
        return {
            'signal': 0,
            'confidence': 0.5,
            'reasoning': "No recent news",
            'data': {}
        }
    
    # Формируем текст новостей
    news_text = "\n".join([f"- {n['title']}: {n['summary'][:200]}..." for n in news])
    
    prompt = f"""You are a crypto news analyst. Based on the following recent news headlines and summaries, determine the overall market sentiment (bullish, bearish, or neutral).

Recent News (last 6 hours):
{news_text}

Instructions:
- Analyze the tone and content of each news item.
- Consider whether news is positive (e.g., adoption, partnerships, upgrades), negative (e.g., hacks, regulations, bans), or neutral.
- Aggregate the sentiment into a single signal:
  - **1 (bullish)** if majority of news are positive.
  - **-1 (bearish)** if majority are negative.
  - **0 (neutral)** if mixed or neutral.
- Provide confidence based on how clear the sentiment is.
- Reasoning should mention key headlines.

Return ONLY a valid JSON:
{{"signal": int, "confidence": float, "reasoning": str}}
signal: -1 bearish, 0 neutral, 1 bullish.
confidence: 0-1.
reasoning: short explanation.
"""

    data = await query_vllm(prompt, log)
    if data is None:
        return {
            'signal': 0,
            'confidence': 0.5,
            'reasoning': "News analysis failed",
            'data': {'news_count': len(news)}
        }
    return {
        'signal': data['signal'],
        'confidence': data['confidence'],
        'reasoning': data['reasoning'],
        'data': {'news': news}
    }

async def main():
    client = RabbitMQClient(RABBITMQ_URL)
    await client.connect()
    logger.info("News worker started, waiting for tasks...")
    await client.start_worker("news_queue", process_task)

if __name__ == "__main__":
    asyncio.run(main())