import asyncio
import pickle
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from collections import defaultdict
import pandas as pd
import numpy as np
from loguru import logger
from config import config
from bybit_client import BybitClient
from strategies import SmartIndicatorStrategy, ChannelDetector, TechnicalIndicatorCalculator, calculate_atr
from market_regime import MarketRegimeDetector
from market_scanner import MarketScanner
from trading_memory import PersistentTradingMemory, TradingMemory
from ai_agents import AITradingOrchestrator, AgentAnalysis
from ai_agents import close_rabbitmq
from telegram_notifier import TelegramNotifier

import ray

class TradingBot:
    def __init__(self):
        self.client = None
        self.scanner = MarketScanner()
        self.strategies = {
            'indicator': SmartIndicatorStrategy(),
        }
        self.regime_detector = MarketRegimeDetector()
        self.channel_detector = ChannelDetector(lookback_period=50)
        self.trading_memory = PersistentTradingMemory()
        self.ai_orchestrator = AITradingOrchestrator()
        self.use_ai = True
        self.running = False
        self.positions = defaultdict(float)
        self.open_positions = {}
        self.trade_history = []
        self.current_symbols = []
        self.TAKER_FEE = 0.00055

        self.last_training_time = None
        self.min_new_experiences_for_training = config.TRAINING_MIN_EXPERIENCES
        self.training_interval_hours = getattr(config, 'TRAINING_INTERVAL_HOURS', 24)
        self.model_path = "models/ppo_trader.zip"
        self.temp_model_path = "models/ppo_trader_new.zip"

        self.trailing_stops = {}
        self.previous_positions = {}
        self.last_recalc_count = 0

        self.telegram = None
        if config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
            self.telegram = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
            logger.info("✅ Telegram notifier initialized")

    def _play_trade_sound(self):
        try:
            print('\a', end='', flush=True)
        except Exception as e:
            logger.debug(f"Sound notification failed: {e}")

    async def initialize(self):
        logger.info("🚀 Initializing Bybit Futures Bot with AI...")
        self.client = BybitClient()
        if hasattr(config, 'TRADE_TOP_VOLUME_COINS') and config.TRADE_TOP_VOLUME_COINS:
            self.current_symbols = await self.scanner.get_dynamic_symbols()
        else:
            self.current_symbols = config.SYMBOLS
        # Принудительно убираем ADA (временное решение)
        self.current_symbols = [s for s in self.current_symbols if s != "ADA/USDT"]
        logger.info(f"📊 Trading {len(self.current_symbols)} symbols")
        for symbol in self.current_symbols:
            data = await self.client.fetch_ohlcv(symbol, '1h', limit=1000)
            if not data.empty:
                self.regime_detector.fit(data)
                regime = self.regime_detector.predict_regime(data)
                logger.info(f"📊 {symbol} regime: {self.regime_detector.get_regime_name(regime)}")
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            logger.info("✅ Ray initialized (for workers)")
        logger.success("✅ Initialization complete")

    def calculate_pnl_with_fees(self, entry_price, exit_price, amount, position_type):
        if position_type == 'long':
            raw_pnl = (exit_price - entry_price) * amount
        else:
            raw_pnl = (entry_price - exit_price) * amount
        entry_fee = entry_price * amount * self.TAKER_FEE
        exit_fee = exit_price * amount * self.TAKER_FEE
        total_fees = entry_fee + exit_fee
        net_pnl = raw_pnl - total_fees
        return {
            'raw_pnl': raw_pnl,
            'fees': total_fees,
            'net_pnl': net_pnl,
            'net_pnl_percent': (net_pnl / (entry_price * amount)) * 100 if entry_price * amount else 0
        }

    async def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        try:
            logger.info(f"START analyze_symbol for {symbol}")
            data = await self.client.fetch_multiple_timeframes(symbol)
            if not data:
                logger.warning(f"No data for {symbol}")
                return None
            logger.info(f"Data fetched for {symbol}, keys: {list(data.keys())}")
            regime = 0
            regime_name = "Unknown"
            if '1h' in data:
                regime = self.regime_detector.predict_regime(data['1h'])
                regime_name = self.regime_detector.get_regime_name(regime)
                logger.info(f"Regime for {symbol}: {regime_name}")
            tf_data = data.get('1m', data.get('5m', data.get('15m')))
            if tf_data is None:
                logger.warning(f"No suitable timeframe data for {symbol}")
                return None
            logger.info(f"Using timeframe data for {symbol}, shape: {tf_data.shape if hasattr(tf_data, 'shape') else 'unknown'}")
            logger.info(f"tf_data type: {type(tf_data)}, shape: {tf_data.shape if hasattr(tf_data, 'shape') else 'no shape'}")

            signals = []
            for strategy in self.strategies.values():
                try:
                    sig = strategy.generate_signal(symbol, tf_data)
                    if sig:
                        signals.append(sig)
                except Exception as e:
                    logger.error(f"Error in generate_signal for {symbol}: {e}", exc_info=True)

            try:
                channel_signal = self.channel_detector.get_trading_signal(tf_data)
                if channel_signal['signal'] != config.SIGNAL_HOLD:
                    class DummySignal:
                        pass
                    ds = DummySignal()
                    ds.signal = channel_signal['signal']
                    ds.confidence = channel_signal['confidence']
                    ds.reasons = [f"Channel {channel_signal['action']}"]
                    ds.price = tf_data['close'].iloc[-1]
                    ds.strategy_name = "ChannelDetector"
                    signals.append(ds)
                    logger.info(f"Channel signal added for {symbol}: {channel_signal['signal']}")
            except Exception as e:
                logger.error(f"Error in channel detector for {symbol}: {e}", exc_info=True)

            ai_analyses = []
            ai_signal = config.SIGNAL_HOLD
            ai_confidence = 0.0
            ai_reasoning = []
            if self.use_ai:
                logger.info(f"Fetching balance for {symbol}")
                balance = await self.client.get_account_balance()
                logger.info(f"Balance fetched for {symbol}: {balance}")
                logger.info(f"Balance fetched, now calling analyze_market for {symbol}")
                logger.info(f"BALANCE FETCHED, NEXT STEP IS analyze_market for {symbol}")
                current_position = self.positions.get(symbol, 0)
                logger.info(f"Current position for {symbol}: {current_position}")
                logger.info(f"Calling analyze_market for {symbol}")
                logger.info(f"🔥🔥🔥 About to call analyze_market for {symbol} 🔥🔥🔥")
                ai_result = await self.ai_orchestrator.analyze_market(
                    symbol,
                    tf_data,
                    balance.get('USDT', 0),
                    regime=regime_name,
                    current_position=current_position
                )
                logger.info(f"AFTER analyze_market call for {symbol}")
                logger.info(f"analyze_market returned for {symbol}: {ai_result}")
                ai_analyses = ai_result.get('analyses', [])
                ai_signal = ai_result['signal']
                ai_confidence = ai_result['confidence']
                ai_reasoning = ai_result['reasoning']
                if ai_signal != config.SIGNAL_HOLD:
                    class AISignal:
                        pass
                    ais = AISignal()
                    ais.signal = ai_signal
                    ais.confidence = ai_confidence
                    ais.reasons = ai_reasoning
                    ais.price = tf_data['close'].iloc[-1]
                    ais.strategy_name = "AI Consensus"
                    signals.append(ais)
                    logger.info(f"AI signal added for {symbol}: {ai_signal}")
            else:
                logger.info("AI disabled")

            if not signals and not self.use_ai:
                logger.warning(f"No signals at all for {symbol}")
                return None

            weighted_signal = sum(s.signal * s.confidence for s in signals) / len(signals)
            avg_confidence = sum(s.confidence for s in signals) / len(signals)
            all_reasons = list(set(r for s in signals for r in (s.reasons if hasattr(s, 'reasons') else [])))[:5]
            price = await self.client.get_last_price(symbol) or signals[0].price

            final_signal = config.SIGNAL_HOLD
            final_confidence = 0.0
            final_reasons = []

            if ai_signal != config.SIGNAL_HOLD:
                final_signal = ai_signal
                final_confidence = ai_confidence
                final_reasons = ai_reasoning
                logger.info(f"🤖 AI signal for {symbol}: {final_signal} with confidence {final_confidence:.2f}")
            else:
                logger.info(f"⏸️ No trade: AI HOLD for {symbol}")
                return None

            if final_signal == config.SIGNAL_HOLD:
                return None

            logger.info(f"Returning analysis for {symbol}")
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'regime': regime_name,
                'final_signal': final_signal,
                'consensus_score': weighted_signal,
                'confidence': final_confidence,
                'price': price,
                'reasons': final_reasons,
                'ai_analyses': ai_analyses if self.use_ai else [],
                'tf_data': tf_data
            }
        except Exception as e:
            logger.error(f"Analyze error {symbol}: {e}", exc_info=True)
            return None

    async def check_daily_loss(self) -> bool:
        if not hasattr(config, 'MAX_DAILY_LOSS_PERCENT'):
            return True
        today = datetime.now().date()
        closes = [t for t in self.trade_history if t['time'].date() == today and t['type'] in ['close_long', 'close_short']]
        if not closes:
            return True
        total_pnl = sum(t.get('net_pnl', 0) for t in closes)
        balance = await self.client.get_account_balance()
        current = balance.get('USDT', 0)
        port_val = current + sum(abs(pos) * (await self.client.get_last_price(sym) or 0) for sym, pos in self.positions.items() if pos != 0)
        initial = port_val - total_pnl
        if initial <= 0:
            return True
        loss_pct = (abs(total_pnl) / initial) * 100 if total_pnl < 0 else 0
        logger.info(f"📊 Daily P&L: ${total_pnl:.2f} ({loss_pct:.2f}%)")
        if loss_pct >= config.MAX_DAILY_LOSS_PERCENT and total_pnl < 0:
            logger.warning("⚠️ Daily loss limit reached")
            if self.telegram:
                await self.telegram.send_message(f"⚠️ Daily loss limit ({loss_pct:.1f}%) reached. Trading paused.")
            return False
        return True

    async def _update_trailing_stops(self):
        if not hasattr(config, 'USE_TRAILING_STOP') or not config.USE_TRAILING_STOP:
            return
        for symbol, position_size in self.positions.items():
            if position_size == 0:
                continue
            current_price = await self.client.get_last_price(symbol)
            if not current_price:
                try:
                    ticker = await self.client.exchange.fetch_ticker(symbol.replace('/USDT', '') + '/USDT:USDT')
                    if ticker and 'last' in ticker:
                        current_price = ticker['last']
                        self.client.last_price_cache[symbol] = current_price
                    else:
                        continue
                except Exception:
                    continue
            is_long = position_size > 0
            entry_info = self.open_positions.get(symbol)
            if not entry_info:
                continue
            entry_price = entry_info['entry_price']
            if is_long:
                profit_pct = (current_price - entry_price) / entry_price * 100
            else:
                profit_pct = (entry_price - current_price) / entry_price * 100
            trailing_data = self.trailing_stops.get(symbol)
            if trailing_data is None:
                if profit_pct >= config.TRAILING_STOP_ACTIVATION:
                    activation_price = current_price
                    side = 'sell' if is_long else 'buy'
                    amount = abs(position_size)
                    order = await self.client.place_trailing_stop(
                        symbol, side, amount, activation_price, config.TRAILING_STOP_DISTANCE
                    )
                    if order:
                        self.trailing_stops[symbol] = {
                            'entry_price': entry_price,
                            'best_price': current_price,
                            'trailing_order_id': order['id']
                        }
                        logger.info(f"🆕 Trailing stop activated for {symbol} at price {current_price}")
            else:
                if (is_long and current_price > trailing_data['best_price']) or \
                   (not is_long and current_price < trailing_data['best_price']):
                    trailing_data['best_price'] = current_price
                    logger.debug(f"Updated best price for {symbol} trailing stop: {current_price}")

    def _save_observation(self, symbol: str, df: pd.DataFrame, action: int, trade_id: int = None):
        try:
            calc = TechnicalIndicatorCalculator()
            df_features = calc.calculate_all(df.tail(100))
            if len(df_features) < 60:
                logger.warning(f"Not enough data for observation of {symbol}")
                return
            window = df_features.iloc[-60:][['close', 'volume', 'rsi', 'macd', 'bb_position', 'volatility']].values
            obs = np.concatenate([window.flatten(), np.zeros(3)]).astype(np.float32)
            obs_blob = pickle.dumps(obs)
            cursor = self.trading_memory.conn.cursor()
            cursor.execute('''
                INSERT INTO training_experiences 
                (timestamp, symbol, observation, action, trade_id, used_in_training)
                VALUES (?, ?, ?, ?, ?, 0)
            ''', (datetime.now().isoformat(), symbol, obs_blob, action, trade_id))
            self.trading_memory.conn.commit()
            logger.debug(f"💾 Saved observation for {symbol}")
        except Exception as e:
            logger.error(f"Failed to save observation: {e}")

    async def _get_replay_buffer(self, limit=config.TRAINING_REPLAY_LIMIT) -> pd.DataFrame:
        cursor = self.trading_memory.conn.cursor()
        cursor.execute('''
            SELECT observation, action, reward, timestamp, symbol
            FROM training_experiences
            WHERE reward IS NOT NULL
            ORDER BY RANDOM()
            LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        if not rows:
            return pd.DataFrame()
        data = []
        for row in rows:
            obs = pickle.loads(row[0])
            obs_list = obs.tolist()
            data.append({
                'observation': obs_list,
                'action': row[1],
                'reward': row[2],
                'timestamp': row[3],
                'symbol': row[4]
            })
        return pd.DataFrame(data)

    async def periodic_training(self):
        while self.running:
            await asyncio.sleep(self.training_interval_hours * 3600)
            if await self._should_train():
                logger.info("🔄 Starting background model fine-tuning...")
                await self._run_training_in_background()

    async def _should_train(self) -> bool:
        cursor = self.trading_memory.conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) FROM training_experiences
            WHERE reward IS NOT NULL
        ''')
        count = cursor.fetchone()[0]
        return count >= self.min_new_experiences_for_training

    async def _run_training_in_background(self):
        replay_data = await self._get_replay_buffer(limit=config.TRAINING_REPLAY_LIMIT)
        if len(replay_data) < self.min_new_experiences_for_training:
            logger.warning("Not enough data in replay buffer, skipping training")
            return
        temp_data_path = "temp_training_data.pkl"
        replay_data.to_pickle(temp_data_path)
        logger.info(f"Saved {len(replay_data)} experiences to {temp_data_path}")
        cmd = [
            "python", "train_rllib.py",
            "--data", temp_data_path,
            "--checkpoint_dir", "models/rllib_checkpoints",
            "--timesteps", str(config.TRAINING_TIMESTEPS)
        ]
        process = await asyncio.create_subprocess_exec(*cmd)
        asyncio.create_task(self._monitor_rllib_training(process, temp_data_path))

    async def _monitor_rllib_training(self, process, temp_data_path):
        try:
            await process.wait()
            if process.returncode == 0:
                logger.success("✅ RLlib training finished successfully")
                self.last_training_time = datetime.now()
            else:
                logger.error(f"❌ RLlib training failed with code {process.returncode}")
        finally:
            if os.path.exists(temp_data_path):
                os.remove(temp_data_path)
                logger.debug(f"Removed temporary file {temp_data_path}")

    async def _calculate_dynamic_tp_sl(self, symbol: str, signal: int, entry_price: float,
                                        df: pd.DataFrame, ai_analyses: List[AgentAnalysis] = None):
        try:
            sl_mult = getattr(config, 'DEFAULT_SL_MULT', 1.5)
            tp_mult = getattr(config, 'DEFAULT_TP_MULT', 3.0)
            atr_series = calculate_atr(df, period=14)
            if atr_series is None or atr_series.empty:
                logger.warning(f"ATR calculation failed for {symbol}, fallback to fixed percentages")
                return self._fallback_tp_sl(signal, entry_price)
            latest_atr = atr_series.iloc[-1]
            if pd.isna(latest_atr) or latest_atr <= 0:
                logger.warning(f"Invalid ATR value ({latest_atr}) for {symbol}, fallback to fixed percentages")
                return self._fallback_tp_sl(signal, entry_price)
            if ai_analyses:
                for a in ai_analyses:
                    if a.agent_name == "Risk Manager" and 'recommended_sl_mult' in a.data:
                        sl_mult = a.data['recommended_sl_mult']
                    if a.agent_name == "Technical Analyst" and 'recommended_tp_mult' in a.data:
                        tp_mult = a.data['recommended_tp_mult']
            if signal == config.SIGNAL_BUY:
                sl_price = entry_price - latest_atr * sl_mult
                tp_price = entry_price + latest_atr * tp_mult
            else:
                sl_price = entry_price + latest_atr * sl_mult
                tp_price = entry_price - latest_atr * tp_mult
            min_gap = max(latest_atr * 0.3, 0.0001)
            if signal == config.SIGNAL_SELL:
                if sl_price <= entry_price:
                    sl_price = entry_price + min_gap
                if tp_price >= entry_price:
                    tp_price = entry_price - min_gap
            else:
                if sl_price >= entry_price:
                    sl_price = entry_price - min_gap
                if tp_price <= entry_price:
                    tp_price = entry_price + min_gap
            min_abs_dist = entry_price * 0.005
            if abs(sl_price - entry_price) < min_abs_dist:
                sl_price = entry_price - min_abs_dist if signal == config.SIGNAL_BUY else entry_price + min_abs_dist
            if abs(tp_price - entry_price) < min_abs_dist:
                tp_price = entry_price + min_abs_dist if signal == config.SIGNAL_BUY else entry_price - min_abs_dist
            sl_price = round(sl_price, 4)
            tp_price = round(tp_price, 4)
            logger.debug(f"Dynamic TP/SL for {symbol}: ATR={latest_atr:.4f}, SL={sl_price}, TP={tp_price}")
            return tp_price, sl_price
        except Exception as e:
            logger.error(f"Error in dynamic TP/SL calculation: {e}")
            return None, None

    def _fallback_tp_sl(self, signal: int, entry_price: float):
        tp_price = None
        sl_price = None
        if hasattr(config, 'TAKE_PROFIT_PERCENT') and config.TAKE_PROFIT_PERCENT > 0:
            if signal == config.SIGNAL_BUY:
                tp_price = entry_price * (1 + config.TAKE_PROFIT_PERCENT / 100)
            else:
                tp_price = entry_price * (1 - config.TAKE_PROFIT_PERCENT / 100)
        if hasattr(config, 'STOP_LOSS_PERCENT') and config.STOP_LOSS_PERCENT > 0:
            if signal == config.SIGNAL_BUY:
                sl_price = entry_price * (1 - config.STOP_LOSS_PERCENT / 100)
            else:
                sl_price = entry_price * (1 + config.STOP_LOSS_PERCENT / 100)
        return tp_price, sl_price

    async def execute_signal(self, analysis: Dict):
        if analysis['final_signal'] == config.SIGNAL_HOLD:
            return
        logger.info(f"Executing signal for {analysis['symbol']}")
        symbol = analysis['symbol']
        signal = analysis['final_signal']
        confidence = analysis['confidence']
        price = analysis['price']
        regime = analysis.get('regime', 'Unknown')
        ai_analyses = analysis.get('ai_analyses', [])
        tf_data = analysis.get('tf_data')

        if regime in ["Trending Bear"] and signal == config.SIGNAL_BUY:
            logger.info(f"⛔ Skipping LONG in {regime}")
            logger.info(f"Signal execution completed for {analysis['symbol']} (skipped LONG)")
            return
        if regime in ["Trending Bull"] and signal == config.SIGNAL_SELL:
            logger.info(f"⛔ Skipping SHORT in {regime}")
            logger.info(f"Signal execution completed for {analysis['symbol']} (skipped SHORT)")
            return

        existing_position = 0
        try:
            positions = await self.client.get_positions()
            for pos_sym, pos_data in positions.items():
                if pos_sym == symbol and pos_data['size'] != 0:
                    existing_position = pos_data['size']
                    self.positions[symbol] = existing_position
                    logger.info(f"📌 Active position: {symbol} {existing_position}")
                    break
        except Exception as e:
            logger.error(f"Position fetch failed: {e}")
            existing_position = self.positions.get(symbol, 0)

        if (signal == config.SIGNAL_SELL and existing_position < 0) or (signal == config.SIGNAL_BUY and existing_position > 0):
            logger.info(f"⏭️ Already have {'SHORT' if existing_position<0 else 'LONG'} on {symbol}, skipping")
            logger.info(f"Signal execution completed for {analysis['symbol']} (already in position)")
            return

        balance = await self.client.get_account_balance()
        usdt_balance = balance.get('USDT', 0)

        base_margin = usdt_balance * (config.MAX_POSITION_SIZE_PERCENT / 100)

        position_multiplier = 1.0
        for a in ai_analyses:
            if a.agent_name == "Risk Manager":
                if 'position_size_multiplier' in a.data:
                    position_multiplier = a.data['position_size_multiplier']
                    logger.info(f"📏 Risk Manager position multiplier: {position_multiplier:.2f}")
                break
        margin = base_margin * confidence * position_multiplier

        if margin < config.MIN_TRADE_AMOUNT_USDT:
            logger.info(f"💰 Margin ${margin:.2f} below minimum ${config.MIN_TRADE_AMOUNT_USDT}")
            logger.info(f"Signal execution completed for {analysis['symbol']} (margin too low)")
            return

        if usdt_balance < margin:
            logger.warning(f"Insufficient balance: have ${usdt_balance:.2f}, need ${margin:.2f}")
            logger.info(f"Signal execution completed for {analysis['symbol']} (insufficient balance)")
            return

        await self.client.set_leverage(symbol, config.LEVERAGE)

        target_notional = margin * config.LEVERAGE
        contracts = round(target_notional / price)

        if contracts == 0:
            logger.info(f"⚠️ Calculated contracts = 0, skipping")
            logger.info(f"Signal execution completed for {analysis['symbol']} (zero contracts)")
            return

        actual_margin = contracts * price / config.LEVERAGE
        if actual_margin > usdt_balance * 1.05:
            logger.warning(f"Actual margin ${actual_margin:.2f} > balance ${usdt_balance:.2f}, skipping")
            logger.info(f"Signal execution completed for {analysis['symbol']} (margin exceeds balance)")
            return

        logger.debug(f"Target notional: ${target_notional:.2f}, contracts: {contracts}, actual margin: ${actual_margin:.2f}")

        if signal == config.SIGNAL_SELL and existing_position > 0:
            logger.info(f"🔚 Closing LONG {symbol}")
            order = await self.client.create_order(symbol, 'sell', existing_position, reduce_only=True)
            if order:
                entry_info = self.open_positions.pop(symbol, None)
                if entry_info:
                    entry_price = entry_info['entry_price']
                    trade_id = entry_info['trade_id']
                    pnl = self.calculate_pnl_with_fees(entry_price, price, existing_position, 'long')
                    logger.info(f"   Raw P&L: ${pnl['raw_pnl']:.2f}, Fees: ${pnl['fees']:.4f}, Net: ${pnl['net_pnl']:.2f}")
                    cursor = self.trading_memory.conn.cursor()
                    cursor.execute('''
                        UPDATE training_experiences SET reward = ? WHERE trade_id = ? AND reward IS NULL
                    ''', (pnl['net_pnl'], trade_id))
                    self.trading_memory.conn.commit()
                    if self.use_ai and ai_analyses:
                        self.ai_orchestrator.update_weights_from_performance(pnl['net_pnl'], ai_analyses)
                self.positions[symbol] = 0
                self.trade_history.append({
                    'symbol': symbol, 'type': 'close_long', 'amount': existing_position,
                    'price': price, 'net_pnl': pnl['net_pnl'] if 'pnl' in locals() else 0,
                    'time': analysis['timestamp']
                })
                self.trailing_stops.pop(symbol, None)
                if self.telegram:
                    await self.telegram.send_trade_notification({
                        'symbol': symbol,
                        'type': 'CLOSE LONG',
                        'price': price,
                        'amount': existing_position,
                        'pnl': pnl['net_pnl'] if 'pnl' in locals() else 0,
                        'confidence': confidence,
                        'regime': regime,
                        'reasons': ', '.join(analysis['reasons'])
                    })
            logger.info(f"Signal execution completed for {analysis['symbol']} (closed long)")
            return

        if signal == config.SIGNAL_BUY and existing_position < 0:
            logger.info(f"🔚 Closing SHORT {symbol}")
            order = await self.client.create_order(symbol, 'buy', abs(existing_position), reduce_only=True)
            if order:
                entry_info = self.open_positions.pop(symbol, None)
                if entry_info:
                    entry_price = entry_info['entry_price']
                    trade_id = entry_info['trade_id']
                    pnl = self.calculate_pnl_with_fees(entry_price, price, abs(existing_position), 'short')
                    logger.info(f"   Raw P&L: ${pnl['raw_pnl']:.2f}, Fees: ${pnl['fees']:.4f}, Net: ${pnl['net_pnl']:.2f}")
                    cursor = self.trading_memory.conn.cursor()
                    cursor.execute('''
                        UPDATE training_experiences SET reward = ? WHERE trade_id = ? AND reward IS NULL
                    ''', (pnl['net_pnl'], trade_id))
                    self.trading_memory.conn.commit()
                    if self.use_ai and ai_analyses:
                        self.ai_orchestrator.update_weights_from_performance(pnl['net_pnl'], ai_analyses)
                self.positions[symbol] = 0
                self.trade_history.append({
                    'symbol': symbol, 'type': 'close_short', 'amount': abs(existing_position),
                    'price': price, 'net_pnl': pnl['net_pnl'] if 'pnl' in locals() else 0,
                    'time': analysis['timestamp']
                })
                self.trailing_stops.pop(symbol, None)
                if self.telegram:
                    await self.telegram.send_trade_notification({
                        'symbol': symbol,
                        'type': 'CLOSE SHORT',
                        'price': price,
                        'amount': abs(existing_position),
                        'pnl': pnl['net_pnl'] if 'pnl' in locals() else 0,
                        'confidence': confidence,
                        'regime': regime,
                        'reasons': ', '.join(analysis['reasons'])
                    })
            logger.info(f"Signal execution completed for {analysis['symbol']} (closed short)")
            return

        bias = 'bullish' if signal == config.SIGNAL_BUY else 'bearish'
        consistency = self.trading_memory.check_consistency(symbol, bias, price)
        if not consistency['consistent']:
            logger.warning(f"⏸️ Memory block: {consistency['reason']} – skipping trade")
            logger.info(f"Signal execution completed for {analysis['symbol']} (memory block)")
            return

        tp_price = None
        sl_price = None
        if hasattr(config, 'USE_TP_SL') and config.USE_TP_SL:
            tp_price, sl_price = await self._calculate_dynamic_tp_sl(
                symbol=symbol,
                signal=signal,
                entry_price=price,
                df=tf_data,
                ai_analyses=ai_analyses
            )
            if tp_price:
                logger.info(f"🎯 Dynamic take profit set at ${tp_price:.2f}")
            if sl_price:
                logger.info(f"🛑 Dynamic stop loss set at ${sl_price:.2f}")

        if signal == config.SIGNAL_BUY and existing_position == 0:
            fee = margin * self.TAKER_FEE
            break_even = (fee * 2 / margin) * 100
            logger.info(f"📈 LONG {symbol} | Margin ${margin:.2f} ({contracts} ct) | Leverage {config.LEVERAGE}x | Conf {confidence:.1%} | Fee ${fee:.4f} | BE {break_even:.3f}%")
            order = await self.client.create_order(
                symbol, 'buy', contracts,
                stop_loss_price=sl_price,
                take_profit_price=tp_price
            )
            if order:
                self._play_trade_sound()
                self.positions[symbol] = contracts
                trade_id = self.trading_memory.store_decision(TradingMemory(
                    symbol=symbol, bias=bias, confidence=confidence,
                    entry_price=price, reason=', '.join(analysis['reasons'])
                ))
                self.open_positions[symbol] = {'entry_price': price, 'trade_id': trade_id}
                self.trade_history.append({
                    'symbol': symbol, 'type': 'open_long', 'amount': contracts,
                    'amount_usdt': margin, 'price': price,
                    'confidence': confidence, 'regime': regime,
                    'reasons': ', '.join(analysis['reasons']),
                    'time': analysis['timestamp'],
                    'trade_id': trade_id
                })
                if ai_analyses:
                    cursor = self.trading_memory.conn.cursor()
                    for a in ai_analyses:
                        cursor.execute('''
                            INSERT INTO agent_predictions (trade_id, agent_name, signal, confidence)
                            VALUES (?, ?, ?, ?)
                        ''', (trade_id, a.agent_name, a.signal, a.confidence))
                    self.trading_memory.conn.commit()
                if tf_data is not None:
                    self._save_observation(symbol, tf_data, config.ACTION_BUY, trade_id)
                self.trailing_stops[symbol] = None
                if self.telegram:
                    await self.telegram.send_trade_open({
                        'symbol': symbol,
                        'type': 'BUY',
                        'price': price,
                        'amount': contracts,
                        'confidence': confidence,
                        'regime': regime,
                        'reasons': ', '.join(analysis['reasons'])
                    })
            logger.info(f"Signal execution completed for {analysis['symbol']} (opened long)")
            return

        elif signal == config.SIGNAL_SELL and existing_position == 0:
            fee = margin * self.TAKER_FEE
            break_even = (fee * 2 / margin) * 100
            logger.info(f"📉 SHORT {symbol} | Margin ${margin:.2f} ({contracts} ct) | Leverage {config.LEVERAGE}x | Conf {confidence:.1%} | Fee ${fee:.4f} | BE {break_even:.3f}%")
            order = await self.client.create_order(
                symbol, 'sell', contracts,
                stop_loss_price=sl_price,
                take_profit_price=tp_price
            )
            if order:
                self._play_trade_sound()
                self.positions[symbol] = -contracts
                trade_id = self.trading_memory.store_decision(TradingMemory(
                    symbol=symbol, bias=bias, confidence=confidence,
                    entry_price=price, reason=', '.join(analysis['reasons'])
                ))
                self.open_positions[symbol] = {'entry_price': price, 'trade_id': trade_id}
                self.trade_history.append({
                    'symbol': symbol, 'type': 'open_short', 'amount': -contracts,
                    'amount_usdt': margin, 'price': price,
                    'confidence': confidence, 'regime': regime,
                    'reasons': ', '.join(analysis['reasons']),
                    'time': analysis['timestamp'],
                    'trade_id': trade_id
                })
                if ai_analyses:
                    cursor = self.trading_memory.conn.cursor()
                    for a in ai_analyses:
                        cursor.execute('''
                            INSERT INTO agent_predictions (trade_id, agent_name, signal, confidence)
                            VALUES (?, ?, ?, ?)
                        ''', (trade_id, a.agent_name, a.signal, a.confidence))
                    self.trading_memory.conn.commit()
                if tf_data is not None:
                    self._save_observation(symbol, tf_data, config.ACTION_SELL, trade_id)
                self.trailing_stops[symbol] = None
                if self.telegram:
                    await self.telegram.send_trade_open({
                        'symbol': symbol,
                        'type': 'SELL',
                        'price': price,
                        'amount': contracts,
                        'confidence': confidence,
                        'regime': regime,
                        'reasons': ', '.join(analysis['reasons'])
                    })
            logger.info(f"Signal execution completed for {analysis['symbol']} (opened short)")
            return

        logger.info(f"Signal execution completed for {analysis['symbol']} (no action)")

    async def run(self):
        await self.initialize()
        logger.info("Starting main loop...")
        self.running = True
        last_refresh = datetime.now()

        asyncio.create_task(self.periodic_training())

        cursor = self.trading_memory.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trades WHERE exit_price IS NOT NULL")
        initial_closed = cursor.fetchone()[0]
        self.last_recalc_count = initial_closed
        logger.info(f"Initialized last_recalc_count from DB: {self.last_recalc_count}")

        while self.running:
            try:
                if not await self.check_daily_loss():
                    await asyncio.sleep(3600)
                    continue

                await self._update_trailing_stops()

                try:
                    positions = await self.client.get_positions()
                    current_positions = {sym: p['size'] for sym, p in positions.items()}
                    logger.debug(f"Current positions from exchange: {current_positions}")
                except Exception as e:
                    logger.error(f"Failed to fetch positions: {e}")
                    current_positions = {}

                for sym, prev_size in list(self.previous_positions.items()):
                    curr_size = current_positions.get(sym, 0)
                    if prev_size != 0 and curr_size == 0:
                        await self._handle_external_close(sym, prev_size)
                self.previous_positions = current_positions.copy()

                for sym, size in current_positions.items():
                    self.positions[sym] = size

                if (datetime.now() - last_refresh).seconds > 6*3600:
                    if hasattr(config, 'TRADE_TOP_VOLUME_COINS') and config.TRADE_TOP_VOLUME_COINS:
                        self.current_symbols = await self.scanner.get_dynamic_symbols()
                        last_refresh = datetime.now()

                for symbol in self.current_symbols:
                    logger.info(f"🔹🔹🔹 TOP OF LOOP for {symbol} 🔹🔹🔹")
                    try:
                        logger.info(f"Processing symbol: {symbol}")
                        analysis = await self.analyze_symbol(symbol)
                        logger.info(f"✅ analyze_symbol returned for {symbol}: {analysis is not None}")
                        if analysis:
                            logger.info(f"🔹 About to log analysis details for {symbol}")
                            logger.info(f"\n{'='*50}\nSymbol: {analysis['symbol']}\nRegime: {analysis['regime']}\nSignal: {analysis['final_signal']} ({analysis['consensus_score']:.2f})\nConfidence: {analysis['confidence']:.2%}\nPrice: ${analysis['price']:.2f}")
                            if analysis['reasons']:
                                for r in analysis['reasons']:
                                    logger.info(f"  - {r}")
                            logger.info(f"🔹 About to execute signal for {symbol}")
                            await self.execute_signal(analysis)
                            logger.info(f"✅ execute_signal completed for {symbol}")
                            logger.info(f"✅ Finished processing {symbol}")
                        else:
                            logger.info(f"⏭️ No analysis for {symbol}")
                    except Exception as e:
                        logger.error(f"❌ Error processing {symbol}: {e}", exc_info=True)
                    logger.info(f"🔹 About to sleep for {symbol}")
                    await asyncio.sleep(1)
                    logger.info(f"✅ Sleep completed for {symbol}")
                    logger.info(f"🔹🔹🔹 BOTTOM OF LOOP for {symbol} 🔹🔹🔹")

                cursor = self.trading_memory.conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM trades WHERE exit_price IS NOT NULL")
                db_closed = cursor.fetchone()[0]
                if db_closed > self.last_recalc_count:
                    logger.info("New closed trades detected, recalculating agent weights")
                    self.ai_orchestrator.recalc_weights_from_history(self.trading_memory, n_last=100)
                    self.ai_orchestrator.print_detailed_agent_stats(self.trading_memory, n_last=100)
                    self.last_recalc_count = db_closed

                logger.info(f"Waiting {config.UPDATE_INTERVAL_SECONDS}s...")
                await asyncio.sleep(config.UPDATE_INTERVAL_SECONDS)

            except KeyboardInterrupt:
                logger.info("Shutdown")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                await asyncio.sleep(10)

    async def _handle_external_close(self, symbol: str, prev_size: float):
        logger.info(f"🔔 Position on {symbol} closed externally (likely TP/SL hit)")
        close_price = await self.client.get_last_price(symbol)
        if close_price is None:
            try:
                ticker = await self.client.exchange.fetch_ticker(symbol.replace('/USDT', '') + '/USDT:USDT')
                close_price = ticker['last'] if ticker else None
            except Exception as e:
                logger.error(f"Failed to fetch ticker for {symbol}: {e}")
                close_price = None

        trade_id = None
        entry_price = None
        if symbol in self.open_positions:
            entry_info = self.open_positions.pop(symbol, None)
            if entry_info:
                entry_price = entry_info['entry_price']
                trade_id = entry_info['trade_id']
        if trade_id is None:
            cursor = self.trading_memory.conn.cursor()
            cursor.execute('''
                SELECT id, entry_price FROM trades
                WHERE symbol = ? AND exit_price IS NULL
                ORDER BY timestamp DESC LIMIT 1
            ''', (symbol,))
            row = cursor.fetchone()
            if row:
                trade_id, entry_price = row
            else:
                logger.warning(f"No open trade found in DB for {symbol}, cannot record external close")
                return

        if close_price is not None and trade_id is not None and entry_price is not None:
            position_type = 'long' if prev_size > 0 else 'short'
            pnl = self.calculate_pnl_with_fees(entry_price, close_price, abs(prev_size), position_type)
            logger.info(f"   External close - Entry: ${entry_price:.4f}, Exit: ${close_price:.4f}, Raw P&L: ${pnl['raw_pnl']:.2f}, Fees: ${pnl['fees']:.4f}, Net: ${pnl['net_pnl']:.2f}")
            self.trading_memory.update_outcome(symbol, close_price, pnl['net_pnl'])
            cursor = self.trading_memory.conn.cursor()
            cursor.execute('''
                UPDATE training_experiences SET reward = ? WHERE trade_id = ? AND reward IS NULL
            ''', (pnl['net_pnl'], trade_id))
            self.trading_memory.conn.commit()
            self.trade_history.append({
                'symbol': symbol,
                'type': f'close_{position_type} (external)',
                'amount': abs(prev_size),
                'price': close_price,
                'net_pnl': pnl['net_pnl'],
                'time': datetime.now()
            })
            if self.telegram:
                await self.telegram.send_trade_notification({
                    'symbol': symbol,
                    'type': f'CLOSE {position_type.upper()} (external)',
                    'price': close_price,
                    'amount': abs(prev_size),
                    'pnl': pnl['net_pnl'],
                    'confidence': 1.0,
                    'regime': 'Unknown',
                    'reasons': 'External TP/SL'
                })

        self.positions[symbol] = 0
        self.trailing_stops.pop(symbol, None)

    async def cleanup(self):
        if self.client:
            await self.client.close()
        await close_rabbitmq()
        if self.telegram:
            await self.telegram.close()
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown")
        logger.info("Cleanup done")

async def main():
    bot = TradingBot()
    try:
        await bot.run()
    finally:
        await bot.cleanup()

if __name__ == "__main__":
    asyncio.run(main())