import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger

@dataclass
class TradingMemory:
    symbol: str
    bias: str
    confidence: float
    entry_price: float
    exit_price: Optional[float] = None
    pnl: float = 0.0
    timestamp: datetime = None
    reason: str = ""

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class PersistentTradingMemory:
    def __init__(self, db_path: str = "trading_memory.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                bias TEXT,
                confidence REAL,
                entry_price REAL,
                exit_price REAL,
                pnl REAL,
                timestamp DATETIME,
                reason TEXT,
                outcome TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bias_history (
                symbol TEXT,
                bias TEXT,
                changed_at DATETIME,
                price REAL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                observation BLOB,
                action INTEGER,
                reward REAL,
                trade_id INTEGER,
                used_in_training INTEGER DEFAULT 0
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_predictions (
                trade_id INTEGER,
                agent_name TEXT,
                signal INTEGER,
                confidence REAL,
                FOREIGN KEY(trade_id) REFERENCES trades(id)
            )
        ''')
        self.conn.commit()

    def store_decision(self, memory: TradingMemory):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO trades (symbol, bias, confidence, entry_price, timestamp, reason)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (memory.symbol, memory.bias, memory.confidence,
              memory.entry_price, memory.timestamp.isoformat(), memory.reason))
        self.conn.commit()
        trade_id = cursor.lastrowid

        # Also record the bias change
        cursor.execute('''
            INSERT INTO bias_history (symbol, bias, changed_at, price)
            VALUES (?, ?, ?, ?)
        ''', (memory.symbol, memory.bias, datetime.now().isoformat(), memory.entry_price))
        self.conn.commit()
        return trade_id

    def update_outcome(self, symbol: str, exit_price: float, pnl: float):
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE trades 
            SET exit_price = ?, pnl = ?, outcome = CASE WHEN ? > 0 THEN 'win' ELSE 'loss' END
            WHERE symbol = ? AND exit_price IS NULL
            ORDER BY timestamp DESC LIMIT 1
        ''', (exit_price, pnl, pnl, symbol))
        self.conn.commit()
        return cursor.lastrowid

    def get_win_rate(self, symbol: str, days: int = 30) -> float:
        cursor = self.conn.cursor()
        cutoff = datetime.now() - timedelta(days=days)
        cursor.execute('''
            SELECT COUNT(*) FROM trades 
            WHERE symbol = ? AND timestamp > ? AND outcome IS NOT NULL
        ''', (symbol, cutoff.isoformat()))
        total = cursor.fetchone()[0]
        if total == 0:
            return 0.5
        cursor.execute('''
            SELECT COUNT(*) FROM trades 
            WHERE symbol = ? AND timestamp > ? AND outcome = 'win'
        ''', (symbol, cutoff.isoformat()))
        wins = cursor.fetchone()[0]
        return wins / total

    def check_consistency(self, symbol: str, proposed_bias: str, current_price: float) -> Dict:
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT bias, changed_at, price FROM bias_history
            WHERE symbol = ? ORDER BY changed_at DESC LIMIT 1
        ''', (symbol,))
        last = cursor.fetchone()
        if last:
            last_bias, last_time, last_price = last
            # last_time is stored as ISO string
            last_time = datetime.fromisoformat(last_time)
            if (datetime.now() - last_time).total_seconds() < 180:
                return {'consistent': False, 'reason': 'Слишком частая смена направления'}
        return {'consistent': True, 'recommendation': 'proceed'}