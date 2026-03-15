import asyncio
import json
import re
import time
import os
import pandas as pd
import numpy as np
import aiohttp
from loguru import logger
from rabbitmq_client import RabbitMQClient, RABBITMQ_URL

_http_session = None

# Кэш для макро-данных
_macro_cache = {
    'data': None,
    'timestamp': 0
}

async def get_http_session():
    global _http_session
    if _http_session is None:
        _http_session = aiohttp.ClientSession()
    return _http_session

async def get_coingecko_global_data() -> dict:
    """Получает глобальные метрики с CoinGecko (бесплатно)"""
    url = "https://api.coingecko.com/api/v3/global"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    data = data['data']
                    return {
                        'btc_dominance': data['market_cap_percentage']['btc'],
                        'eth_dominance': data['market_cap_percentage'].get('eth', 17.2),
                        'total_market_cap': data['total_market_cap']['usd'],
                        'total_volume_24h': data['total_volume']['usd'],
                        'market_trend': 'sideways',  # можно не использовать
                    }
                else:
                    logger.warning(f"CoinGecko returned {resp.status}, using defaults")
    except Exception as e:
        logger.warning(f"CoinGecko fetch failed: {e}")
    # fallback
    return {
        'btc_dominance': 58.5,
        'eth_dominance': 17.2,
        'total_market_cap': 2.4e12,
        'total_volume_24h': 80e9,
        'market_trend': 'sideways',
    }

async def get_fear_greed() -> int:
    """Получает Fear & Greed Index с alternative.me (бесплатно)"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.alternative.me/fng/?limit=1", timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return int(data['data'][0]['value'])
    except Exception as e:
        logger.warning(f"Fear & Greed fetch failed: {e}")
    return 45  # значение по умолчанию

async def get_macro_data() -> dict:
    global _macro_cache

    current_time = time.time()
    if _macro_cache['data'] and (current_time - _macro_cache['timestamp']) < 300:
        return _macro_cache['data']

    # Получаем данные параллельно
    coingecko_task = asyncio.create_task(get_coingecko_global_data())
    fng_task = asyncio.create_task(get_fear_greed())

    coingecko_data = await coingecko_task
    fear_greed = await fng_task

    macro_data = {
        'btc_dominance': coingecko_data['btc_dominance'],
        'total_market_cap': coingecko_data['total_market_cap'],
        'fear_greed_index': fear_greed,
        'avg_volatility': 2.1,  # заглушка, можно не использовать
        'market_trend': coingecko_data.get('market_trend', 'sideways'),
    }

    logger.info(f"✅ Macro data updated: BTC dom={macro_data['btc_dominance']}%, F&G={macro_data['fear_greed_index']}")

    _macro_cache['data'] = macro_data
    _macro_cache['timestamp'] = current_time
    return macro_data

async def query_vllm(prompt: str, log, max_tokens: int = 256, retries: int = 3) -> dict:
    url = "http://localhost:8000/v1/completions"
    payload = {
        "model": "/mnt/ssd_combined/models/llama-3.3-70b-awq",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "top_p": 0.9
    }
    session = await get_http_session()
    log.debug(f"Prompt: {prompt}")
    for attempt in range(retries):
        try:
            start = time.time()
            async with session.post(url, json=payload, timeout=90) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    log.error(f"vLLM error {resp.status}: {error_text}")
                    if 500 <= resp.status < 600 and attempt < retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return None
                result = await resp.json()
                text = result['choices'][0]['text'].strip()
                for match in re.finditer(r'\{.*?\}', text, re.DOTALL):
                    candidate = match.group()
                    try:
                        data = json.loads(candidate)
                        if all(k in data for k in ('signal', 'confidence', 'reasoning')):
                            elapsed = time.time() - start
                            log.debug(f"vLLM parsed response: {data} (took {elapsed:.2f}s)")
                            return data
                    except json.JSONDecodeError:
                        continue
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    candidate = match.group()
                    try:
                        data = json.loads(candidate)
                        if all(k in data for k in ('signal', 'confidence', 'reasoning')):
                            elapsed = time.time() - start
                            log.debug(f"vLLM parsed response: {data} (took {elapsed:.2f}s)")
                            return data
                    except json.JSONDecodeError:
                        pass
                log.error(f"Could not extract valid JSON from vLLM response: {text}")
                return None
        except asyncio.TimeoutError:
            log.warning(f"vLLM request timeout (attempt {attempt+1}/{retries})")
        except Exception as e:
            log.error(f"vLLM request failed: {e}")
        await asyncio.sleep(2 ** attempt)
    return None

def _compute_enhanced_sentiment(df: pd.DataFrame) -> dict:
    """Расширенный расчёт сентимент-метрик"""
    if df.empty or len(df) < 20:
        return {
            'price_change': 0.0,
            'volatility': 0.0,
            'volatility_trend': 'neutral',
            'price_trend': 'neutral',
        }
    close = df['close'].values
    price_change = (close[-1] / close[0] - 1) * 100
    returns = np.diff(close) / close[:-1]
    volatility = np.std(returns) * 100 if len(returns) > 0 else 0.0

    # Тренд волатильности (за последние 10 периодов)
    if len(returns) >= 10:
        vol_rolling = [np.std(returns[i-10:i]) for i in range(10, len(returns)+1)]
        if len(vol_rolling) >= 2:
            vol_slope = np.polyfit(range(len(vol_rolling)), vol_rolling, 1)[0]
            if vol_slope > 0.01:
                volatility_trend = 'increasing'
            elif vol_slope < -0.01:
                volatility_trend = 'decreasing'
            else:
                volatility_trend = 'neutral'
        else:
            volatility_trend = 'neutral'
    else:
        volatility_trend = 'neutral'

    # Тренд цены (линейная регрессия за последние 10 свечей)
    if len(close) >= 10:
        price_slope = np.polyfit(range(10), close[-10:], 1)[0]
        if price_slope > close.mean() * 0.001:
            price_trend = 'uptrend'
        elif price_slope < -close.mean() * 0.001:
            price_trend = 'downtrend'
        else:
            price_trend = 'sideways'
    else:
        price_trend = 'sideways'

    recent_prices = close[-10:].tolist() if len(close) >= 10 else close.tolist()

    return {
        'price_change': price_change,
        'volatility': volatility,
        'volatility_trend': volatility_trend,
        'price_trend': price_trend,
        'recent_prices': [float(p) for p in recent_prices],
    }

async def process_task(task: dict) -> dict:
    correlation_id = task.get('correlation_id', 'unknown')
    log = logger.bind(correlation_id=correlation_id)

    df_dict = task['df']
    regime = task.get('regime', 'Unknown')
    macro = await get_macro_data()
    
    df = pd.DataFrame(df_dict)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

    sent = _compute_enhanced_sentiment(df)
    prices_str = ', '.join([f"${p:.2f}" for p in sent['recent_prices']])

    prompt = f"""You are an expert sentiment analyst in a live crypto trading environment. Your task is to gauge market sentiment based on recent price action, volatility, and macro context, then output a directional bias. The market is often ranging with low volatility, so you need to detect subtle shifts.

**Market Context:**
- Regime: {regime}
- BTC Dominance: {macro['btc_dominance']}%
- Fear & Greed Index: {macro['fear_greed_index']}
- Overall Market Trend: {macro.get('market_trend', 'sideways')}

**Recent Price Data (last 10 closes):**
{prices_str}

**Current Metrics:**
- Current Price: ${df['close'].iloc[-1] if not df.empty else 0:.4f}
- Price change over last period: {sent['price_change']:.2f}%
- Price trend (last 10 periods): {sent['price_trend']} (uptrend/downtrend/sideways)
- Volatility (20-period): {sent['volatility']:.4f}% (trend: {sent['volatility_trend']})

**Instructions:**
- Analyze the data: look for signs of accumulation/distribution, shifts in momentum, and reactions to macro factors.
- Consider Fear & Greed extremes: extreme fear (<25) often signals a buying opportunity, extreme greed (>75) a selling opportunity (contrarian view).
- BTC dominance: rising dominance may signal altcoin weakness, falling dominance may signal altcoin strength.
- Volatility trends: increasing volatility can precede breakouts, decreasing volatility suggests consolidation.
- Price trend: even subtle changes (e.g., higher lows in a range) can indicate shifting sentiment.
- Then decide on sentiment: **bullish (1)**, **neutral (0)**, or **bearish (-1)**.
- Assign a confidence score between 0.0 and 1.0. The confidence should reflect how clear the signals are. Even if the signal is weak (e.g., 0.3-0.4), it's acceptable as long as you see a slight edge.
- Provide a short reasoning (1-2 sentences) explaining your logic.

**Examples:**
- If price is making higher lows within a range and volatility is compressing, you might be cautiously bullish (confidence 0.5).
- If BTC dominance is rising sharply and alts are underperforming, you might be bearish on alts (confidence 0.6).
- If Fear & Greed is at 20 (extreme fear) and price is at support, you might be bullish (confidence 0.7).

**Output strictly in JSON format:**
{{"signal": int, "confidence": float, "reasoning": str}}
signal: -1 (bearish), 0 (neutral), 1 (bullish)
confidence: 0.0 to 1.0
reasoning: short explanation
"""

    data = await query_vllm(prompt, log, max_tokens=256)
    if data is None:
        return {
            'signal': 0,
            'confidence': 0.5,
            'reasoning': "vLLM timeout/error",
            'data': sent
        }
    return {
        'signal': data['signal'],
        'confidence': data['confidence'],
        'reasoning': data['reasoning'],
        'data': sent
    }

async def main():
    client = RabbitMQClient(RABBITMQ_URL)
    await client.connect()
    logger.info("Sentiment worker (vLLM) started, waiting for tasks...")
    await client.start_worker("sentiment_queue", process_task)

if __name__ == "__main__":
    asyncio.run(main())