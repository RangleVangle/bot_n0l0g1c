import ccxt
import ccxt.pro as ccxtpro
import pandas as pd
import asyncio
from datetime import datetime
from typing import Optional, Dict
from loguru import logger
from config import config

class BybitClient:
    def __init__(self):
        self.exchange = ccxt.bybit({
            'enableRateLimit': True,
            'timeout': 30000,
            'apiKey': config.BYBIT_API_KEY if not config.DRY_RUN else '',
            'secret': config.BYBIT_SECRET_KEY if not config.DRY_RUN else '',
            'options': {
                'defaultType': 'linear',
                'adjustForTimeDifference': True,
                'unifiedAccount': True,
            }
        })
        
        self.ws_exchange = ccxtpro.bybit({
            'enableRateLimit': True,
            'apiKey': config.BYBIT_API_KEY if not config.DRY_RUN else '',
            'secret': config.BYBIT_SECRET_KEY if not config.DRY_RUN else '',
            'options': {
                'defaultType': 'linear',
                'unifiedAccount': True,
            }
        })
        
        self.last_price_cache = {}
        self.positions_cache = {}
        
        if config.DRY_RUN:
            logger.info("🌐 FUTURES MODE - DRY RUN")
        else:
            logger.info("🔑 FUTURES MODE - LIVE TRADING")

    async def close(self):
        try:
            await self.ws_exchange.close()
            logger.info("Connections closed")
        except Exception as e:
            logger.error(f"Error closing: {e}")

    def _convert_symbol(self, symbol: str) -> str:
        if ':' not in symbol and '/USDT' in symbol:
            base = symbol.replace('/USDT', '')
            return f"{base}/USDT:USDT"
        return symbol

    def fetch_ohlcv_sync(self, symbol: str, timeframe: str = '1h', limit: int = 500) -> pd.DataFrame:
        try:
            futures_symbol = self._convert_symbol(symbol)
            data = self.exchange.fetch_ohlcv(futures_symbol, timeframe, limit=limit)
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            if not df.empty:
                self.last_price_cache[symbol] = float(df['close'].iloc[-1])
            logger.debug(f"Fetched {len(df)} candles for {symbol} ({timeframe})")
            return df
        except Exception as e:
            logger.error(f"OHLCV error {symbol}: {e}")
            return pd.DataFrame()

    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 500) -> pd.DataFrame:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.fetch_ohlcv_sync, symbol, timeframe, limit)

    async def fetch_multiple_timeframes(self, symbol: str) -> Dict[str, pd.DataFrame]:
        data = {}
        for tf in config.TIMEFRAMES:
            try:
                df = await self.fetch_ohlcv(symbol, tf, limit=200)
                if not df.empty:
                    data[tf] = df
            except Exception as e:
                logger.error(f"Error fetching {tf} for {symbol}: {e}")
            await asyncio.sleep(0.5)
        return data

    async def get_account_balance(self) -> dict:
        logger.info("Entering get_account_balance")
        if config.DRY_RUN:
            return {'USDT': config.SIMULATED_BALANCE_USDT}
        if not config.BYBIT_API_KEY or not config.BYBIT_SECRET_KEY:
            return {'USDT': 0}
        try:
            loop = asyncio.get_event_loop()
            balance = await loop.run_in_executor(None, self.exchange.fetch_balance)
            if 'info' in balance and 'result' in balance['info']:
                for item in balance['info']['result'].get('list', []):
                    if item.get('coin') == 'USDT':
                        available = float(item.get('availableBalance', 0))
                        if available > 0:
                            logger.info(f"💰 Futures available: ${available}")
                            return {'USDT': available}
            if 'USDT' in balance.get('free', {}):
                free_bal = float(balance['free']['USDT'])
                logger.info(f"💰 Free USDT: ${free_bal}")
                return {'USDT': free_bal}
            if 'total' in balance and 'USDT' in balance['total']:
                total_bal = float(balance['total']['USDT'])
                logger.info(f"💰 Total USDT (may include spot): ${total_bal}")
                return {'USDT': total_bal}
            logger.error("Could not find USDT balance")
            return {'USDT': 0}
        except Exception as e:
            logger.error(f"Balance fetch error: {e}")
            return {'USDT': 0}
        finally:
            logger.info("Exiting get_account_balance")

    async def get_positions(self) -> Dict:
        if config.DRY_RUN:
            return {}
        try:
            loop = asyncio.get_event_loop()
            positions = await loop.run_in_executor(None, self.exchange.fetch_positions)
            result = {}
            for pos in positions:
                size = pos.get('contracts')
                if size is not None:
                    try:
                        size_float = float(size)
                        if size_float != 0:
                            symbol = pos['symbol'].replace(':USDT', '')
                            result[symbol] = {
                                'size': size_float,
                                'entry_price': float(pos.get('entryPrice') or 0),
                                'pnl': float(pos.get('unrealizedPnl') or 0),
                                'percentage': float(pos.get('percentage') or 0)
                            }
                    except (TypeError, ValueError):
                        continue
            return result
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return {}

    async def set_leverage(self, symbol: str, leverage: int):
        try:
            futures_symbol = self._convert_symbol(symbol)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.exchange.set_leverage(leverage, futures_symbol)
            )
            logger.info(f"⚙️ Leverage for {symbol} set to {leverage}x")
            return result
        except Exception as e:
            if "leverage not modified" in str(e):
                logger.debug(f"Leverage for {symbol} already {leverage}x")
            else:
                logger.warning(f"Could not set leverage for {symbol}: {e}")
            return None

    async def create_order(self, symbol: str, side: str, amount: float, price: float = None,
                           reduce_only: bool = False,
                           stop_loss_price: float = None,
                           take_profit_price: float = None):
        action = "CLOSE" if reduce_only else "OPEN"
        logger.info(f"🔧 ORDER: {action} {side.upper()} {amount} {symbol}")
        futures_symbol = self._convert_symbol(symbol)
        if config.DRY_RUN:
            logger.info(f"🔬 DRY RUN: {side.upper()} {amount} {symbol} {'(CLOSE)' if reduce_only else ''}")
            return {'id': 'dry_run', 'status': 'ok'}
        if not config.BYBIT_API_KEY or not config.BYBIT_SECRET_KEY:
            logger.error("No API keys")
            return None
        try:
            order_type = 'market' if price is None else 'limit'
            params = {}
            if reduce_only:
                params['reduce_only'] = True
            if stop_loss_price is not None:
                params['stopLoss'] = stop_loss_price
            if take_profit_price is not None:
                params['takeProfit'] = take_profit_price
            loop = asyncio.get_event_loop()
            order = await loop.run_in_executor(
                None,
                lambda: self.exchange.create_order(futures_symbol, order_type, side, amount, price, params)
            )
            logger.success(f"✅ Order placed: {order['id']}")
            return order
        except Exception as e:
            logger.error(f"Order failed: {e}")
            return None

    async def place_trailing_stop(self, symbol: str, side: str, amount: float, activation_price: float, trailing_dist: float):
        """
        Places a trailing stop order on Bybit for linear futures.
        - trailing_dist: percentage distance (e.g., 0.3 for 0.3%) – will be converted to absolute price.
        """
        futures_symbol = self._convert_symbol(symbol)
        try:
            # Convert percentage to absolute price distance
            price_distance = activation_price * trailing_dist / 100.0

            params = {
                'triggerPrice': activation_price,      # activation price
                'trailingAmount': price_distance,      # absolute distance in price units
                'reduceOnly': True,                    # must be true for closing orders
            }

            loop = asyncio.get_event_loop()
            order = await loop.run_in_executor(
                None,
                lambda: self.exchange.create_order(futures_symbol, 'market', side, amount, None, params)
            )
            logger.info(f"🔄 Trailing stop placed for {symbol} at activation {activation_price}, distance {trailing_dist}% (abs {price_distance:.4f})")
            return order
        except Exception as e:
            logger.error(f"Failed to place trailing stop for {symbol}: {e}")
            return None

    async def get_last_price(self, symbol: str) -> Optional[float]:
        """Get last cached price for a symbol."""
        return self.last_price_cache.get(symbol)

    async def test_connection(self) -> bool:
        try:
            ticker = self.exchange.fetch_ticker('BTC/USDT:USDT')
            if ticker and 'last' in ticker:
                logger.info(f"✅ Connected. BTC: ${ticker['last']}")
                return True
            return False
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False