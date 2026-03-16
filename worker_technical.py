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

def _compute_enhanced_indicators(df: pd.DataFrame) -> dict:
    """Расширенный расчёт индикаторов для technical аналитика"""
    if df.empty or len(df) < 50:
        return {
            'price': 0,
            'rsi': 50,
            'macd': 0,
            'volume_ratio': 1.0,
            'sma20': 0,
            'sma50': 0,
            'bb_upper': 0,
            'bb_lower': 0,
            'bb_position': 0.5,
            'recent_prices': [],
            'rsi_trend': 'neutral',
            'volume_trend': 'neutral',
        }
    
    close = df['close'].values
    volume = df['volume'].values
    
    # RSI
    deltas = np.diff(close)
    gain = np.mean(deltas[deltas > 0]) if any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if any(deltas < 0) else 0
    if loss == 0:
        rsi = 100 if gain > 0 else 50
    else:
        rs = gain / loss
        rsi = 100 - 100 / (1 + rs)
    
    # MACD
    ema12 = pd.Series(close).ewm(span=12).mean().iloc[-1]
    ema26 = pd.Series(close).ewm(span=26).mean().iloc[-1]
    macd = ema12 - ema26
    
    # Volume ratio
    volume_sma = pd.Series(volume).rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
    volume_ratio = volume[-1] / volume_sma if volume_sma > 0 else 1.0
    
    # SMA
    sma20 = pd.Series(close).rolling(20).mean().iloc[-1]
    sma50 = pd.Series(close).rolling(50).mean().iloc[-1] if len(close) >= 50 else sma20
    
    # Bollinger Bands (20,2)
    sma20_series = pd.Series(close).rolling(20).mean()
    std20 = pd.Series(close).rolling(20).std()
    bb_upper = (sma20_series + 2 * std20).iloc[-1]
    bb_lower = (sma20_series - 2 * std20).iloc[-1]
    bb_position = (close[-1] - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
    
    # Последние 10 цен
    recent_prices = close[-10:].tolist() if len(close) >= 10 else close.tolist()
    
    # Тренд RSI (за последние 5 свечей)
    if len(close) >= 15:
        rsi_values = []
        for i in range(-5, 0):
            window = close[i-14:i+1]
            if len(window) >= 14:
                deltas_w = np.diff(window)
                gain_w = np.mean(deltas_w[deltas_w > 0]) if any(deltas_w > 0) else 0
                loss_w = -np.mean(deltas_w[deltas_w < 0]) if any(deltas_w < 0) else 0
                if loss_w == 0:
                    rsi_w = 100 if gain_w > 0 else 50
                else:
                    rs_w = gain_w / loss_w
                    rsi_w = 100 - 100 / (1 + rs_w)
                rsi_values.append(rsi_w)
        if len(rsi_values) >= 3:
            rsi_slope = np.polyfit(range(len(rsi_values)), rsi_values, 1)[0]
            if rsi_slope > 2:
                rsi_trend = 'rising'
            elif rsi_slope < -2:
                rsi_trend = 'falling'
            else:
                rsi_trend = 'neutral'
        else:
            rsi_trend = 'neutral'
    else:
        rsi_trend = 'neutral'
    
    # Тренд объёма
    if len(volume) >= 10:
        vol_slope = np.polyfit(range(10), volume[-10:], 1)[0]
        if vol_slope > volume_sma * 0.05:
            volume_trend = 'increasing'
        elif vol_slope < -volume_sma * 0.05:
            volume_trend = 'decreasing'
        else:
            volume_trend = 'neutral'
    else:
        volume_trend = 'neutral'
    
    return {
        'price': float(close[-1]),
        'rsi': float(rsi),
        'macd': float(macd),
        'volume_ratio': float(volume_ratio),
        'sma20': float(sma20),
        'sma50': float(sma50),
        'bb_upper': float(bb_upper),
        'bb_lower': float(bb_lower),
        'bb_position': float(bb_position),
        'recent_prices': [float(p) for p in recent_prices],
        'rsi_trend': rsi_trend,
        'volume_trend': volume_trend,
    }

async def process_task(task: dict) -> dict:
    correlation_id = task.get('correlation_id', 'unknown')
    log = logger.bind(correlation_id=correlation_id)
    log.info(f"🔥 Technical received task for {task.get('symbol', 'unknown')}")

    df_dict = task['df']
    regime = task.get('regime', 'Unknown')
    macro = await get_macro_data()
    
    df = pd.DataFrame(df_dict)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

    ind = _compute_enhanced_indicators(df)

    prices_str = ', '.join([f"${p:.2f}" for p in ind['recent_prices']])

    prompt = f"""You are an experienced scalper in a low-volatility ranging market. Your goal is to find small 0.3-1.0% moves. 
Analyze the data and decide: BUY (1), SELL (-1), or HOLD (0). Even subtle signals can be traded.

Market context: {regime}, BTC dom: {macro['btc_dominance']}%, F&G: {macro['fear_greed_index']}.

Current data (last 10 candles):
{prices_str}
- Price: ${ind['price']:.4f}
- RSI(14): {ind['rsi']:.1f} (trend: {ind['rsi_trend']})
- MACD: {ind['macd']:.4f}
- Volume ratio: {ind['volume_ratio']:.2f}x (trend: {ind['volume_trend']})
- BB position: {ind['bb_position']:.2f} (0=lower,1=upper)
- Support (lower BB): ${ind['bb_lower']:.4f}
- Resistance (upper BB): ${ind['bb_upper']:.4f}

Consider:
- If price is near support (BB ≤0.3) and RSI is not falling → possible bounce (BUY).
- If price is near resistance (BB ≥0.7) and RSI is not rising → possible pullback (SELL).
- Look for minor divergences: price making lower low but RSI higher low → bullish.
- Volume spikes even without price movement can signal accumulation/distribution.
- If MACD is flattening after a move, it may precede reversal.

Assign confidence 0.3-0.8 based on how clear the setup is. Even low confidence (0.3-0.4) is acceptable if you see a slight edge.

Output JSON with fields: signal (int), confidence (float), reasoning (str).
"""

    data = await query_vllm(prompt, log, max_tokens=256)
    if data is None:
        return {
            'signal': 0,
            'confidence': 0.5,
            'reasoning': "vLLM timeout/error",
            'data': ind
        }
    return {
        'signal': data['signal'],
        'confidence': data['confidence'],
        'reasoning': data['reasoning'],
        'data': ind
    }

async def main():
    client = RabbitMQClient(RABBITMQ_URL)
    await client.connect()
    logger.info("Technical worker (vLLM) started, waiting for tasks...")
    await client.start_worker("technical_queue", process_task)

if __name__ == "__main__":
    asyncio.run(main())