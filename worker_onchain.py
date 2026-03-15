import os
from dotenv import load_dotenv
load_dotenv()
import asyncio
import json
import time
import aiohttp
from loguru import logger
from rabbitmq_client import RabbitMQClient, RABBITMQ_URL

# Кэш для on-chain данных (обновляем раз в час)
_cache = {
    'data': {},
    'timestamp': 0
}

async def fetch_dexpaprika_metrics() -> dict:
    """Получает глобальные метрики с DexPaprika (бесплатно, без ключа)"""
    result = {}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.dexpaprika.com/v1/tickers", timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result['btc_dominance'] = data.get('btc_dominance', 58.5)
                    result['eth_dominance'] = data.get('eth_dominance', 17.2)
                    result['total_market_cap'] = data.get('total_market_cap_usd', 2.4e12)
                    result['total_volume_24h'] = data.get('total_volume_usd_24h', 80e9)
                    # DexPaprika не даёт DeFi/stablecoin объёмы, оставим заглушки
                    result['defi_volume'] = 5e9
                    result['stablecoin_volume'] = 60e9
                    logger.info("✅ DexPaprika data fetched successfully")
                else:
                    logger.warning(f"DexPaprika returned {resp.status}, using defaults")
    except Exception as e:
        logger.warning(f"DexPaprika fetch failed: {e}")
    
    return result

async def fetch_coingecko_metrics() -> dict:
    """Получает глобальные метрики с CoinGecko (бесплатно, без ключа)"""
    result = {}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.coingecko.com/api/v3/global", timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    data = data['data']
                    result['btc_dominance'] = data['market_cap_percentage']['btc']
                    result['eth_dominance'] = data['market_cap_percentage'].get('eth', 17.2)
                    result['total_market_cap'] = data['total_market_cap']['usd']
                    result['total_volume_24h'] = data['total_volume']['usd']
                    # CoinGecko тоже не даёт DeFi/stablecoin объёмы
                    result['defi_volume'] = 5e9
                    result['stablecoin_volume'] = 60e9
                    logger.info("✅ CoinGecko data fetched successfully")
                else:
                    logger.warning(f"CoinGecko returned {resp.status}")
    except Exception as e:
        logger.warning(f"CoinGecko fetch failed: {e}")
    return result

async def get_onchain_data() -> dict:
    """Получает on-chain данные с кэшированием (1 час)"""
    global _cache
    now = time.time()
    if _cache['data'] and (now - _cache['timestamp']) < 3600:
        return _cache['data']
    
    # Пробуем DexPaprika
    data = await fetch_dexpaprika_metrics()
    
    # Если не удалось получить основные метрики, пробуем CoinGecko
    if not data.get('btc_dominance'):
        data = await fetch_coingecko_metrics()
    
    # Если оба источника не сработали, используем значения по умолчанию
    if not data.get('btc_dominance'):
        data = {
            'btc_dominance': 58.5,
            'eth_dominance': 17.2,
            'total_market_cap': 2.4e12,
            'total_volume_24h': 80e9,
            'defi_volume': 5e9,
            'stablecoin_volume': 60e9
        }
        logger.warning("Both DexPaprika and CoinGecko failed, using defaults")
    
    _cache['data'] = data
    _cache['timestamp'] = now
    return data

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
                async with session.post(url, json=payload, timeout=90) as resp:
                    if resp.status != 200:
                        log.error(f"vLLM error {resp.status}, attempt {attempt+1}/{retries}")
                        if attempt < retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return None
                    result = await resp.json()
                    text = result['choices'][0]['text'].strip()
                    import re
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

    onchain = await get_onchain_data()
    
    prompt = f"""You are an on-chain analyst specializing in cryptocurrency markets. Based on the following global blockchain metrics, provide a trading signal (BUY/SELL/HOLD) and reasoning.

**Current On-Chain Metrics:**
- BTC Dominance: {onchain.get('btc_dominance', 'N/A')}%
- ETH Dominance: {onchain.get('eth_dominance', 'N/A')}%
- Total Market Cap: ${onchain.get('total_market_cap', 0):,.0f}
- 24h Total Volume: ${onchain.get('total_volume_24h', 0):,.0f}
- 24h DeFi Volume: ${onchain.get('defi_volume', 0):,.0f}
- 24h Stablecoin Volume: ${onchain.get('stablecoin_volume', 0):,.0f}

**Instructions:**
- Analyze the data: 
  - High BTC dominance often precedes altcoin weakness; low dominance suggests altcoin season.
  - Increasing total volume indicates market activity; decreasing volume may signal consolidation.
  - High DeFi/stablecoin volume can indicate risk-on/off sentiment.
- Decide on a market-wide bias:
  - **1 (bullish)**: if metrics suggest strong overall market health (rising volume, balanced dominance, etc.)
  - **-1 (bearish)**: if metrics suggest weakness (falling volume, extreme BTC dominance, etc.)
  - **0 (neutral)**: if mixed.
- Provide reasoning.

Return ONLY a valid JSON object:
{{"signal": int, "confidence": float, "reasoning": str}}
signal: -1 bearish, 0 neutral, 1 bullish.
confidence: 0-1.
reasoning: short professional explanation.
"""

    data = await query_vllm(prompt, log)
    if data is None:
        return {
            'signal': 0,
            'confidence': 0.5,
            'reasoning': "On-chain data unavailable",
            'data': onchain
        }
    return {
        'signal': data['signal'],
        'confidence': data['confidence'],
        'reasoning': data['reasoning'],
        'data': onchain
    }

async def main():
    client = RabbitMQClient(RABBITMQ_URL)
    await client.connect()
    logger.info("On-chain worker started, waiting for tasks...")
    await client.start_worker("onchain_queue", process_task)

if __name__ == "__main__":
    asyncio.run(main())