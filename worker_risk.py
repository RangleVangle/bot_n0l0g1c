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
                        if all(k in data for k in ('signal', 'confidence', 'reasoning', 'position_size_multiplier')):
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
                        if all(k in data for k in ('signal', 'confidence', 'reasoning', 'position_size_multiplier')):
                            elapsed = time.time() - start
                            log.debug(f"vLLM parsed response: {data} (took {elapsed:.2f}s)")
                            return data
                    except json.JSONDecodeError:
                        pass
                log.error(f"Could not extract valid JSON with required fields from vLLM response: {text}")
                return None
        except asyncio.TimeoutError:
            log.warning(f"vLLM request timeout (attempt {attempt+1}/{retries})")
        except Exception as e:
            log.error(f"vLLM request failed: {e}")
        await asyncio.sleep(2 ** attempt)
    return None

async def process_task(task: dict) -> dict:
    correlation_id = task.get('correlation_id', 'unknown')
    log = logger.bind(correlation_id=correlation_id)

    df_dict = task['df']
    balance = task.get('balance', 0)
    regime = task.get('regime', 'Unknown')
    current_position = task.get('current_position', 0)
    macro = await get_macro_data()
    
    df = pd.DataFrame(df_dict)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

    if df.empty or len(df) < 20:
        volatility = 0.0
        vol_trend = 'neutral'
    else:
        close = df['close'].values
        returns = np.diff(close) / close[:-1]
        volatility = float(np.std(returns) * 100)

        # Тренд волатильности
        if len(returns) >= 10:
            vol_rolling = [np.std(returns[i-10:i]) for i in range(10, len(returns)+1)]
            if len(vol_rolling) >= 2:
                vol_slope = np.polyfit(range(len(vol_rolling)), vol_rolling, 1)[0]
                if vol_slope > 0.01:
                    vol_trend = 'increasing'
                elif vol_slope < -0.01:
                    vol_trend = 'decreasing'
                else:
                    vol_trend = 'neutral'
            else:
                vol_trend = 'neutral'
        else:
            vol_trend = 'neutral'

    portfolio_risk = abs(current_position) * volatility / 100

    prompt = f"""You are a professional risk manager for a crypto trading fund. Your role is to decide whether the current market conditions are safe enough to open a new position, and to suggest an appropriate position size multiplier based on risk.

**Market Context:**
- Market Regime: {regime}
- BTC Dominance: {macro['btc_dominance']}%
- Fear & Greed Index: {macro['fear_greed_index']}
- Average Market Volatility: {macro['avg_volatility']}%

**Current Risk Metrics:**
- Recent volatility (20 periods): {volatility:.4f}% (trend: {vol_trend})
- Account balance: ${balance:.2f}
- Current position size (contracts): {current_position} (positive = long, negative = short, 0 = no position)
- Estimated portfolio risk from current position: ${portfolio_risk:.2f}

**Instructions:**
- Assess the overall market risk based on volatility, regime, and existing position.
- Compare current volatility to average market volatility – is it significantly higher? If volatility is increasing, be cautious.
- Consider Fear & Greed: extreme values often precede reversals, increasing risk.
- Take into account existing position: if already have a large position, additional risk may be too high.
- Then output:
  - **signal**: -1 (avoid trading), 0 (trade with caution), 1 (trade allowed).
  - **position_size_multiplier**: a float between 0.5 and 1.5. 
        * 0.5 means use half the normal size (high risk).
        * 1.0 means normal size.
        * 1.5 means increased size (very low risk, high confidence).
  - **confidence**: how certain you are about the risk assessment (0-1).
  - **reasoning**: short professional explanation.

**Examples:**

Example 1 (Avoid trading, high risk):
Input: Volatility 5.2% (increasing), Balance $1000, Regime: Trending Bear, Current position: 0
Reasoning: "Extreme and increasing volatility. Very high risk environment. Avoid trading. Multiplier 0.5 if forced, but better to avoid."
Output: {{"signal": -1, "confidence": 0.9, "position_size_multiplier": 0.5, "reasoning": "Extreme and increasing volatility – risk too high for new positions."}}

Example 2 (Trade allowed, low risk):
Input: Volatility 1.1% (stable), Balance $10000, Regime: Ranging, Current position: 0
Reasoning: "Low, stable volatility. Very low risk. Can trade with normal or slightly increased size."
Output: {{"signal": 1, "confidence": 0.9, "position_size_multiplier": 1.2, "reasoning": "Low risk environment – trading allowed with increased size."}}

Example 3 (Caution, moderate risk):
Input: Volatility 2.5% (stable), Balance $5000, Regime: Trending Bull, Current position: +50 contracts
Reasoning: "Moderate risk and existing long position. Use reduced size."
Output: {{"signal": 0, "confidence": 0.7, "position_size_multiplier": 0.7, "reasoning": "Moderate risk and existing position – trade with caution, reduce size."}}

**Now evaluate:**
- Volatility: {volatility:.4f}% (trend: {vol_trend})
- Balance: ${balance:.2f}
- Current position: {current_position}
- Regime: {regime}

Return ONLY a valid JSON:
{{"signal": int, "confidence": float, "position_size_multiplier": float, "reasoning": str}}
"""

    data = await query_vllm(prompt, log, max_tokens=256)
    if data is None:
        return {
            'signal': 0,
            'confidence': 0.5,
            'reasoning': "vLLM timeout/error",
            'data': {'volatility': volatility, 'position_size_multiplier': 1.0}
        }
    multiplier = data.get('position_size_multiplier', 1.0)
    multiplier = max(0.5, min(1.5, multiplier))
    return {
        'signal': data['signal'],
        'confidence': data['confidence'],
        'reasoning': data['reasoning'],
        'data': {
            'volatility': volatility,
            'position_size_multiplier': multiplier
        }
    }

async def main():
    client = RabbitMQClient(RABBITMQ_URL)
    await client.connect()
    logger.info("Risk worker (vLLM) started, waiting for tasks...")
    await client.start_worker("risk_queue", process_task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()