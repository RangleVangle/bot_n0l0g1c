import asyncio
import json
import time
import numpy as np
from loguru import logger
from rabbitmq_client import RabbitMQClient, RABBITMQ_URL
import ccxt.pro as ccxtpro
from typing import Optional

class OrderFlowAnalyzer:
    def __init__(self):
        self.exchange = None
        self.orderbooks = {}
        self.lock = asyncio.Lock()

    def _convert_symbol(self, symbol: str) -> str:
        if ':' not in symbol and '/USDT' in symbol:
            base = symbol.replace('/USDT', '')
            return f"{base}/USDT:USDT"
        return symbol

    async def ensure_exchange(self):
        if self.exchange is None:
            self.exchange = ccxtpro.bybit({
                'enableRateLimit': True,
                'options': {'defaultType': 'linear'}
            })
        return self.exchange

    async def fetch_order_book(self, symbol: str, timeout: float = 5.0) -> Optional[dict]:
        try:
            exchange = await self.ensure_exchange()
            market_symbol = self._convert_symbol(symbol)
            logger.info(f"Fetching order book for {symbol} -> {market_symbol}")  
            ob = await asyncio.wait_for(
                exchange.watch_order_book(market_symbol),
                timeout=timeout
            )
            return ob
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching order book for {symbol}")
            return None
        except Exception as e:
            logger.error(f"Order book error for {symbol}: {e}")
            return None

    def compute_imbalance(self, ob, levels=10):
        """Вычисляет дисбаланс ликвидности на первых levels уровнях"""
        bids = ob['bids'][:levels]
        asks = ob['asks'][:levels]
        bid_vol = sum([b[1] for b in bids])
        ask_vol = sum([a[1] for a in asks])
        total = bid_vol + ask_vol
        if total == 0:
            return 0
        imbalance = (bid_vol - ask_vol) / total
        return imbalance

    def detect_whale(self, ob, threshold=100.0):
        """Ищет крупный ордер (более threshold базовой валюты)"""
        for bid in ob['bids']:
            if bid[1] > threshold:
                return 'bid', bid[0], bid[1]
        for ask in ob['asks']:
            if ask[1] > threshold:
                return 'ask', ask[0], ask[1]
        return None, None, None

    async def analyze(self, symbol: str) -> dict:
        ob = await self.fetch_order_book(symbol)
        if not ob:
            return {'signal': 0, 'confidence': 0, 'reasoning': 'No order book'}

        imbalance = self.compute_imbalance(ob)
        whale_side, whale_price, whale_vol = self.detect_whale(ob)

        reasoning = []
        signal = 0
        confidence = 0.0

        # Логика сигнала на основе дисбаланса
        if imbalance > 0.3:
            signal = 1
            confidence = min(abs(imbalance), 0.9)
            reasoning.append(f"Strong bid imbalance: {imbalance:.2f}")
        elif imbalance < -0.3:
            signal = -1
            confidence = min(abs(imbalance), 0.9)
            reasoning.append(f"Strong ask imbalance: {imbalance:.2f}")
        else:
            signal = 0
            confidence = 0.3
            reasoning.append("Neutral order book")

        # Усиление сигнала, если есть кит
        if whale_side == 'bid' and signal == 1:
            confidence = min(confidence + 0.2, 1.0)
            reasoning.append(f"Large bid {whale_vol:.1f} at {whale_price:.4f}")
        elif whale_side == 'ask' and signal == -1:
            confidence = min(confidence + 0.2, 1.0)
            reasoning.append(f"Large ask {whale_vol:.1f} at {whale_price:.4f}")
        elif whale_side:
            # Кит против сигнала — уменьшаем уверенность
            confidence = max(confidence - 0.1, 0.1)
            reasoning.append(f"Whale on opposite side")

        return {
            'signal': signal,
            'confidence': round(confidence, 2),
            'reasoning': '; '.join(reasoning),
            'data': {
                'imbalance': imbalance,
                'bid_volume': sum(b[1] for b in ob['bids'][:10]),
                'ask_volume': sum(a[1] for a in ob['asks'][:10]),
                'whale': whale_side
            }
        }

    async def close(self):
        if self.exchange:
            await self.exchange.close()
            self.exchange = None

# Глобальный экземпляр (для переиспользования WebSocket)
_orderflow_analyzer = None

async def get_analyzer():
    global _orderflow_analyzer
    if _orderflow_analyzer is None:
        _orderflow_analyzer = OrderFlowAnalyzer()
    return _orderflow_analyzer
    def _convert_symbol(self, symbol: str) -> str:
        """Преобразует символ из формата BTC/USDT в формат BTC/USDT:USDT для Bybit."""
        if ':' not in symbol and '/USDT' in symbol:
            base = symbol.replace('/USDT', '')
            return f"{base}/USDT:USDT"
        return symbol

async def process_task(task: dict) -> dict:
    correlation_id = task.get('correlation_id', 'unknown')
    log = logger.bind(correlation_id=correlation_id)
    symbol = task.get('symbol')
    if not symbol:
        return {'signal': 0, 'confidence': 0, 'reasoning': 'No symbol provided'}

    log.info(f"🔥 OrderFlow processing {symbol}")
    analyzer = await get_analyzer()
    try:
        result = await analyzer.analyze(symbol)
        log.debug(f"OrderFlow result for {symbol}: {result}")
        return result
    except Exception as e:
        log.error(f"Error in orderflow analysis for {symbol}: {e}")
        return {'signal': 0, 'confidence': 0, 'reasoning': f'Error: {e}'}

async def main():
    client = RabbitMQClient(RABBITMQ_URL)
    await client.connect()
    logger.info("OrderFlow worker started, waiting for tasks...")
    try:
        await client.start_worker("orderflow_queue", process_task)
    finally:
        # Закрываем соединение с биржей при остановке
        if _orderflow_analyzer:
            await _orderflow_analyzer.close()
        await client.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("OrderFlow worker stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")