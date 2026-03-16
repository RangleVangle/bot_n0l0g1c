import asyncio
import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
from loguru import logger
from rabbitmq_client import RabbitMQClient
from config import config

_rabbitmq_client: Optional[RabbitMQClient] = None

async def get_rabbitmq_client() -> RabbitMQClient:
    global _rabbitmq_client
    if _rabbitmq_client is None:
        _rabbitmq_client = RabbitMQClient()
        await _rabbitmq_client.connect()
    return _rabbitmq_client

async def close_rabbitmq():
    global _rabbitmq_client
    if _rabbitmq_client:
        await _rabbitmq_client.close()
        _rabbitmq_client = None



@dataclass
class AgentAnalysis:
    agent_name: str
    signal: int
    confidence: float
    reasoning: str
    data: Dict

class RemoteAgent:
    def __init__(self, name: str, queue_name: str):
        self.name = name
        self.queue_name = queue_name

    async def analyze(self, symbol: str, df: pd.DataFrame, balance: float = None, regime: str = None, current_position: float = None) -> AgentAnalysis:
        logger.info(f"Preparing task for {self.name} with symbol {symbol}")
        df_dict = df.tail(200).to_dict(orient='records')
        task = {'df': df_dict, 'symbol': symbol}
        if balance is not None:
            task['balance'] = balance
        if regime is not None:
            task['regime'] = regime
        if current_position is not None:
            task['current_position'] = current_position

        client = await get_rabbitmq_client()
        logger.info(f"Sending task to {self.queue_name} for {symbol}")
        result = await client.publish_task(self.queue_name, task)
        if result is None:
            logger.warning(f"No result from {self.name} for {symbol}")
            return AgentAnalysis(self.name, config.SIGNAL_HOLD, 0.5, "Remote agent timeout/error", {})
        logger.info(f"Received result from {self.name} for {symbol}")
        return AgentAnalysis(
            agent_name=self.name,
            signal=result['signal'],
            confidence=result['confidence'],
            reasoning=result['reasoning'],
            data=result.get('data', {})
        )

class TechnicalAnalyst(RemoteAgent):
    def __init__(self):
        super().__init__("Technical Analyst", "technical_queue")

class SentimentAnalyst(RemoteAgent):
    def __init__(self):
        super().__init__("Sentiment Analyst", "sentiment_queue")

class RiskManager(RemoteAgent):
    def __init__(self):
        super().__init__("Risk Manager", "risk_queue")

class RLAgent(RemoteAgent):
    def __init__(self):
        super().__init__("RL Master", "rl_queue")

class TrianglePatternAgent(RemoteAgent):
    def __init__(self):
        super().__init__("Triangle Scanner", "triangle_queue")

class RetestAgent(RemoteAgent):
    def __init__(self):
        super().__init__("Retest Scanner", "retest_queue")

class OnChainAgent(RemoteAgent):
    def __init__(self):
        super().__init__("On-Chain Analyst", "onchain_queue")

class NewsAgent(RemoteAgent):
    def __init__(self):
        super().__init__("News Analyst", "news_queue")

class OrderFlowAgent(RemoteAgent):
    def __init__(self):
        super().__init__("Order Flow Analyst", "orderflow_queue")

class AITradingOrchestrator:
    def __init__(self, weights_file="agent_weights.json"):
        self.agents = [
            TechnicalAnalyst(),
            SentimentAnalyst(),      # раскомментируйте позже
            RiskManager(),
            # RLAgent(),
            TrianglePatternAgent(),
            RetestAgent(),
            OnChainAgent(),
            NewsAgent(),
            OrderFlowAgent()            # новый агент
        ]
        self.agent_weights = {
            'Technical Analyst': 0.20,
            'Sentiment Analyst': 0.15,
            'Risk Manager': 0.10,
            'RL Master': 0.05,
            'Triangle Scanner': 0.05,
            'Retest Scanner': 0.05,
            'On-Chain Analyst': 0.15,
            'News Analyst': 0.10,
            'Order Flow Analyst': 0.15
        }
        self.weights_file = weights_file
        self._load_weights()
        logger.info(f"Initialized AITradingOrchestrator with weights: {self.agent_weights}")

    def _load_weights(self):
        if os.path.exists(self.weights_file):
            with open(self.weights_file, 'r') as f:
                saved = json.load(f)
                for name, w in saved.items():
                    if name in self.agent_weights:
                        self.agent_weights[name] = w
            logger.info(f"Loaded weights from {self.weights_file}: {self.agent_weights}")
        else:
            logger.info(f"Weights file {self.weights_file} not found, using defaults")

    def _save_weights(self):
        with open(self.weights_file, 'w') as f:
            json.dump(self.agent_weights, f, indent=2)
        logger.info(f"Saved weights to {self.weights_file}: {self.agent_weights}")

    def recalc_weights_from_history(self, memory, n_last=100):
        logger.info(f"🔁 recalc_weights_from_history called with n_last={n_last}")
        cursor = memory.conn.cursor()
        cursor.execute('''
            SELECT t.id, t.pnl FROM trades t
            WHERE t.exit_price IS NOT NULL
            ORDER BY t.timestamp DESC
            LIMIT ?
        ''', (n_last,))
        trades = cursor.fetchall()
        logger.info(f"Fetched {len(trades)} closed trades from DB")
        if len(trades) < 2:
            logger.warning(f"Only {len(trades)} trades, skipping recalculation")
            return
        agent_stats = {}
        for trade_id, pnl in trades:
            outcome = 'win' if pnl > 0 else 'loss'
            cursor.execute('''
                SELECT agent_name, signal, confidence FROM agent_predictions
                WHERE trade_id = ?
            ''', (trade_id,))
            preds = cursor.fetchall()
            for agent_name, signal, conf in preds:
                if agent_name not in agent_stats:
                    agent_stats[agent_name] = {
                        'wins': 0, 'losses': 0,
                        'total_conf_win': 0, 'total_conf_loss': 0,
                        'total_trades': 0
                    }
                agent_stats[agent_name]['total_trades'] += 1
                if outcome == 'win':
                    agent_stats[agent_name]['wins'] += 1
                    agent_stats[agent_name]['total_conf_win'] += conf
                else:
                    agent_stats[agent_name]['losses'] += 1
                    agent_stats[agent_name]['total_conf_loss'] += conf
        logger.info(f"Agent stats collected: {agent_stats}")
        alpha = 0.7
        for agent_name, stats in agent_stats.items():
            total = stats['wins'] + stats['losses']
            if total == 0:
                continue
            win_rate = stats['wins'] / total
            avg_conf_win = stats['total_conf_win'] / stats['wins'] if stats['wins'] > 0 else 0.5
            avg_conf_loss = stats['total_conf_loss'] / stats['losses'] if stats['losses'] > 0 else 0.5
            raw_score = win_rate + (avg_conf_win - avg_conf_loss)
            normalized_score = (raw_score + 1) / 2
            normalized_score = max(0.1, min(0.9, normalized_score))
            old_weight = self.agent_weights.get(agent_name, 0.33)
            new_weight = (1 - alpha) * old_weight + alpha * normalized_score
            new_weight = max(0.05, min(1.0, new_weight))
            logger.info(f"📊 Agent {agent_name}: old={old_weight:.3f}, new={new_weight:.3f}, "
                        f"wr={win_rate:.2f}, conf_win={avg_conf_win:.2f}, conf_loss={avg_conf_loss:.2f}, "
                        f"raw={raw_score:.2f}, norm={normalized_score:.2f}")
            self.agent_weights[agent_name] = new_weight
        self._save_weights()
        logger.info(f"✅ Weights after recalculation: {self.agent_weights}")
        logger.info(f"Weights recalculated for {len(self.agents)} agents")

    async def analyze_market(self, symbol: str, df: pd.DataFrame, balance: float, regime: str = None, current_position: float = None) -> Dict:
        logger.info(f"ENTER analyze_market for {symbol}")
        tasks = []
        logger.info(f"🔍 Analyzing {symbol} with {len(self.agents)} agents: {[agent.name for agent in self.agents]}")
        for agent in self.agents:
            logger.debug(f"Creating task for {agent.name} ({agent.queue_name})")
            if agent.name == "Risk Manager":
                coro = agent.analyze(symbol, df, balance, regime=regime, current_position=current_position)
            else:
                coro = agent.analyze(symbol, df, balance, regime=regime)
            tasks.append(asyncio.create_task(coro))  # <-- явное создание задачи

        done, pending = await asyncio.wait(tasks, timeout=45, return_when=asyncio.ALL_COMPLETED)

        if pending:
            logger.error(f"⏰ Timeout: {len(pending)} tasks still pending for {symbol}")
            for task in pending:
                task.cancel()
            analyses = []
            for task in done:
                try:
                    result = task.result()
                    analyses.append(result)
                except Exception as e:
                    logger.error(f"Task raised exception: {e}")
                    analyses.append(e)
        else:
            analyses = [task.result() for task in done]
            logger.debug(f"gather completed for {symbol}, got {len(analyses)} results")

        valid = [a for a in analyses if isinstance(a, AgentAnalysis)]
        if not valid:
            logger.warning("No valid agent analyses received")
            return {'signal': config.SIGNAL_HOLD, 'confidence': 0, 'reasoning': []}

        buy_power = 0.0
        sell_power = 0.0
        total_weight = 0.0
        all_reasoning = []

        for a in valid:
            w = self.agent_weights.get(a.agent_name, 0.33)
            weighted_conf = a.confidence * w
            if a.signal == config.SIGNAL_BUY:
                buy_power += weighted_conf
            elif a.signal == config.SIGNAL_SELL:
                sell_power += weighted_conf
            total_weight += w
            all_reasoning.append(f"{a.agent_name}: {a.reasoning}")
            logger.debug(f"Agent {a.agent_name}: signal={a.signal}, conf={a.confidence:.2f}, weight={w:.3f}")

        threshold = config.VOTE_THRESHOLD
        final_signal = config.SIGNAL_HOLD
        confidence = 0.0

        if buy_power > sell_power and buy_power > threshold:
            final_signal = config.SIGNAL_BUY
            confidence = min(buy_power / total_weight, 1.0)
        elif sell_power > buy_power and sell_power > threshold:
            final_signal = config.SIGNAL_SELL
            confidence = min(sell_power / total_weight, 1.0)
        else:
            final_signal = config.SIGNAL_HOLD
            confidence = max(buy_power, sell_power) / total_weight if total_weight else 0
            logger.debug(f"Vote below threshold: buy={buy_power:.2f}, sell={sell_power:.2f}, threshold={threshold}")

        logger.info(f"Market analysis result: buy_power={buy_power:.2f}, sell_power={sell_power:.2f}, "
                    f"signal={final_signal}, confidence={confidence:.2f}")
        return {
            'signal': final_signal,
            'consensus_score': (buy_power - sell_power) / total_weight if total_weight else 0,
            'confidence': confidence,
            'reasoning': all_reasoning,
            'analyses': valid
        }

    def update_weights_from_performance(self, pnl: float, analyses: List[AgentAnalysis]):
        logger.info(f"💰 update_weights_from_performance called with pnl={pnl:.2f}")
        for a in analyses:
            w = self.agent_weights.get(a.agent_name, 0.33)
            old_w = w
            if pnl > 0 and a.signal != config.SIGNAL_HOLD:
                w = w * 1.05
                logger.debug(f"Agent {a.agent_name} rewarded: {old_w:.3f} -> {w:.3f}")
            elif pnl < 0 and a.signal != config.SIGNAL_HOLD:
                w = w * 0.95
                logger.debug(f"Agent {a.agent_name} punished: {old_w:.3f} -> {w:.3f}")
            else:
                logger.debug(f"Agent {a.agent_name} no change (hold signal or zero pnl)")
            w = max(0.05, min(1.0, w))
            self.agent_weights[a.agent_name] = w
        self._save_weights()
        logger.info(f"✅ Weights after performance update: {self.agent_weights}")

    def print_detailed_agent_stats(self, memory, n_last=100):
        logger.info(f"📈 Detailed agent statistics for last {n_last} trades:")
        cursor = memory.conn.cursor()
        cursor.execute('''
            SELECT id, pnl FROM trades
            WHERE exit_price IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (n_last,))
        trades = cursor.fetchall()
        if not trades:
            logger.warning("No closed trades found.")
            return
        trade_ids = [t[0] for t in trades]
        placeholders = ','.join(['?'] * len(trade_ids))
        cursor.execute(f'''
            SELECT agent_name, signal, confidence, trade_id
            FROM agent_predictions
            WHERE trade_id IN ({placeholders})
        ''', trade_ids)
        predictions = cursor.fetchall()
        trade_pnl = {t[0]: t[1] for t in trades}
        stats = {}
        for agent_name, signal, conf, tid in predictions:
            if signal == 0:
                continue
            if agent_name not in stats:
                stats[agent_name] = {
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_conf_win': 0.0,
                    'total_conf_loss': 0.0,
                    'total_pnl': 0.0,
                    'total_pnl_win': 0.0,
                    'total_pnl_loss': 0.0,
                }
            s = stats[agent_name]
            s['trades'] += 1
            pnl = trade_pnl[tid]
            s['total_pnl'] += pnl
            if pnl > 0:
                s['wins'] += 1
                s['total_conf_win'] += conf
                s['total_pnl_win'] += pnl
            else:
                s['losses'] += 1
                s['total_conf_loss'] += conf
                s['total_pnl_loss'] += pnl
        header = f"{'Agent':<25} {'Trades':>6} {'Wins':>5} {'Losses':>6} {'WR%':>5} {'AvgConfWin':>10} {'AvgConfLoss':>11} {'Total PnL':>10} {'Avg PnL':>8}"
        logger.info(header)
        logger.info("-" * len(header))
        for agent_name, s in stats.items():
            win_rate = s['wins'] / s['trades'] if s['trades'] > 0 else 0
            avg_conf_win = s['total_conf_win'] / s['wins'] if s['wins'] > 0 else 0
            avg_conf_loss = s['total_conf_loss'] / s['losses'] if s['losses'] > 0 else 0
            avg_pnl = s['total_pnl'] / s['trades'] if s['trades'] > 0 else 0
            logger.info(f"{agent_name:<25} {s['trades']:>6} {s['wins']:>5} {s['losses']:>6} "
                        f"{win_rate*100:>5.1f}% {avg_conf_win:>10.2f} {avg_conf_loss:>11.2f} "
                        f"{s['total_pnl']:>10.2f} {avg_pnl:>8.2f}")
        total_trades = len(trades)
        total_pnl = sum(t[1] for t in trades)
        logger.info("-" * len(header))
        logger.info(f"{'ALL TRADES':<25} {total_trades:>6} {'-':>5} {'-':>6} {'-':>5} {'-':>10} {'-':>11} {total_pnl:>10.2f} {'-':>8}")