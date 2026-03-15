import ccxt
import asyncio
from typing import List, Dict
from loguru import logger
from config import config
from strategies import calculate_atr  # импортируем функцию расчёта ATR
import pandas as pd

class MarketScanner:
    def __init__(self):
        self.exchange = ccxt.bybit({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',
            }
        })
        self.all_symbols = []
        self.top_symbols = []

    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 20) -> pd.DataFrame:
        """Получает свечи для расчёта ATR (синхронно, но в отдельном потоке)."""
        try:
            loop = asyncio.get_event_loop()
            futures_symbol = self._convert_symbol(symbol)
            data = await loop.run_in_executor(None, self.exchange.fetch_ohlcv, futures_symbol, timeframe, limit)
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.debug(f"Failed to fetch OHLCV for {symbol}: {e}")
            return pd.DataFrame()

    def _convert_symbol(self, symbol: str) -> str:
        if ':' not in symbol and '/USDT' in symbol:
            base = symbol.replace('/USDT', '')
            return f"{base}/USDT:USDT"
        return symbol

    async def fetch_all_futures_pairs(self) -> List[str]:
        try:
            loop = asyncio.get_event_loop()
            markets = await loop.run_in_executor(None, self.exchange.load_markets)
            futures_pairs = []
            for symbol, market in markets.items():
                if (market.get('linear') and 
                    market.get('quote') == 'USDT' and 
                    market.get('active')):
                    simple = symbol.replace(':USDT', '')
                    futures_pairs.append(simple)
            logger.info(f"📊 Found {len(futures_pairs)} USDT perpetual futures")
            self.all_symbols = futures_pairs
            return futures_pairs
        except Exception as e:
            logger.error(f"Error fetching futures markets: {e}")
            return []

    async def get_top_volume_futures(self, limit: int = 10) -> List[str]:
        try:
            all_futures = await self.fetch_all_futures_pairs()
            if not all_futures:
                logger.warning("No futures pairs found, using fallback list")
                filtered = [s for s in config.FALLBACK_SYMBOLS if s not in config.EXCLUDED_SYMBOLS]
                return filtered[:limit]

            loop = asyncio.get_event_loop()
            tickers = await loop.run_in_executor(None, self.exchange.fetch_tickers)

            futures_with_metrics = []
            for simple in all_futures:
                if simple in config.EXCLUDED_SYMBOLS:
                    continue
                bybit_symbol = f"{simple}:USDT"
                if bybit_symbol in tickers:
                    vol = tickers[bybit_symbol].get('quoteVolume', 0)
                    if vol and vol > 0:
                        # Получаем ATR для оценки волатильности
                        df = await self.fetch_ohlcv(simple, timeframe='1h', limit=20)
                        atr = 0
                        if not df.empty:
                            atr_series = calculate_atr(df, period=14)
                            if not atr_series.empty:
                                atr = atr_series.iloc[-1]
                        # Используем произведение объёма и ATR как метрику
                        # Если ATR = 0, используем только объём (но это редко)
                        score = vol * atr if atr > 0 else vol
                        futures_with_metrics.append({
                            'symbol': simple,
                            'volume': vol,
                            'atr': atr,
                            'score': score
                        })

            # Сортируем по убыванию скора (объём * ATR)
            futures_with_metrics.sort(key=lambda x: x['score'], reverse=True)
            top = [item['symbol'] for item in futures_with_metrics[:limit]]
            
            if top:
                logger.info(f"🔥 Top {limit} volatile+volume coins: {top}")
                # Выводим для отладки значения скора
                debug_info = [(item['symbol'], f"{item['score']:.2e}") for item in futures_with_metrics[:5]]
                logger.debug(f"Top scores: {debug_info}")
                return top
            else:
                logger.warning("No volume data, using fallback")
                filtered = [s for s in config.FALLBACK_SYMBOLS if s not in config.EXCLUDED_SYMBOLS]
                return filtered[:limit]
        except Exception as e:
            logger.error(f"Error getting top volumes: {e}")
            filtered = [s for s in config.FALLBACK_SYMBOLS if s not in config.EXCLUDED_SYMBOLS]
            return filtered[:limit]

    async def get_dynamic_symbols(self) -> List[str]:
        if hasattr(config, 'TRADE_TOP_VOLUME_COINS') and config.TRADE_TOP_VOLUME_COINS:
            symbols = await self.get_top_volume_futures(config.TOP_COINS_COUNT)
        else:
            symbols = config.SYMBOLS
            symbols = [s for s in symbols if s not in config.EXCLUDED_SYMBOLS]
        if not symbols:
            symbols = config.FALLBACK_SYMBOLS
            symbols = [s for s in symbols if s not in config.EXCLUDED_SYMBOLS]
        logger.info(f"🎯 Trading {len(symbols)} symbols (excluded: {config.EXCLUDED_SYMBOLS}): {symbols}")
        return symbols