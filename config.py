import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self):
        self.BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
        self.BYBIT_SECRET_KEY = os.getenv("BYBIT_SECRET_KEY", "")
        
        # ========== ВКЛЮЧАЕМ СКАНЕР ==========
        self.TRADE_TOP_VOLUME_COINS = False
        self.TOP_COINS_COUNT = 15
        
        # ========== ИСКЛЮЧЁННЫЕ ТОКЕНЫ ==========
        self.EXCLUDED_SYMBOLS = ["XRP/USDT", "BTC/USDT", "ETH/USDT","SOL/USDT"]
        
        # ========== РУЧНОЙ СПИСОК ==========
        self.SYMBOLS = [
            'DOGE/USDT', 'ADA/USDT', 'LINK/USDT',
            'DOT/USDT', 'AVAX/USDT', 'NEAR/USDT', 'ATOM/USDT',
            'ALGO/USDT', 'SAND/USDT', 'MANA/USDT', 'AXS/USDT',
            'APE/USDT', 'GALA/USDT', 'ROSE/USDT', 'KSM/USDT',
            'CRV/USDT', '1INCH/USDT', 'UNI/USDT', 'AAVE/USDT',
            'COMP/USDT', 'SNX/USDT', 'SUSHI/USDT', 'YFI/USDT',
            'ZRX/USDT', 'BAT/USDT', 'ENJ/USDT', 'CHZ/USDT',
            'ARB/USDT', 'OP/USDT', 'LDO/USDT', 'APT/USDT',
            'SUI/USDT', 'SEI/USDT', 'TIA/USDT', 'INJ/USDT',
            'BLUR/USDT', 'WLD/USDT', 'PYTH/USDT', 'JUP/USDT',
            'JTO/USDT', 'ORDI/USDT', '1000PEPE/USDT', '1000BONK/USDT',
        ]
        self.FALLBACK_SYMBOLS = self.SYMBOLS
        
        # Таймфреймы
        self.TIMEFRAMES = ["1m", "5m", "15m", "1h"]
        self.BASE_QUOTE = "USDT"
        
        # Тип торговли – фьючерсы
        self.MARKET_TYPE = 'linear'
        self.LEVERAGE = 3
        self.POSITION_MODE = 'one-way'
        
        # Размер позиции
        self.MAX_POSITION_SIZE_PERCENT = 20          # ⬅️ уменьшено с 30%
        self.MIN_TRADE_AMOUNT_USDT = 5
        
        # Риск-менеджмент
        self.MAX_DAILY_LOSS_PERCENT = 90
        self.MIN_SIGNAL_STRENGTH = 3.0
        self.MIN_CONFIDENCE = 0.30          # ⬅️ немного снижено (было 0.5)

        # Volatility filter for indicator strategy
        self.VOLATILITY_THRESHOLD = 0.001  # 0.1% от цены (было 0.005 = 0.5%)
        
        self.TRAINING_INTERVAL_HOURS = 3  # или 24

        # Параметры обучения RL
        self.TRAINING_MIN_EXPERIENCES = 10      # минимум опытов для старта обучения
        self.TRAINING_REPLAY_LIMIT = 2000       # размер выборки из буфера
        self.TRAINING_TIMESTEPS = 50000         # сколько шагов обучаться за раз

        # Симулированный баланс
        self.SIMULATED_BALANCE_USDT = 10000
        
        # Настройки данных
        self.LOOKBACK_DAYS = 30
        self.UPDATE_INTERVAL_SECONDS = 5
        
        # Режим
        self.DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

        # ========== TP/SL Settings ==========
        self.USE_TP_SL = True                      # включаем TP/SL (динамические)
        self.TAKE_PROFIT_PERCENT = 0.95            # запасной вариант (если ATR не сработает)
        self.STOP_LOSS_PERCENT = 0.51              # запасной вариант (если ATR не сработает)
        self.USE_TRAILING_STOP = True               # ⬅️ включено
        self.TRAILING_STOP_ACTIVATION = 1.0         # активация после 1% прибыли
        self.TRAILING_STOP_DISTANCE = 0.5           # отступ 0.5% от максимума
        
        # ========== Dynamic TP/SL Settings (на основе ATR) ==========
        self.DEFAULT_SL_MULT = 2.0                  # ⬅️ уменьшено (было 2.5)
        self.DEFAULT_TP_MULT = 5.0                  # ⬅️ увеличено (было 4.0)
        # ============================================================

        # ========== Triangle Detector Settings ==========
        self.TRIANGLE_LOOKBACK = 100
        self.TRIANGLE_MIN_TOUCHES = 3
        self.TRIANGLE_VOLUME_WINDOW = 20
        self.TRIANGLE_MAX_DEVIATION = 0.02

        # ========== Action and Signal Constants ==========
        self.ACTION_HOLD = 0
        self.ACTION_BUY = 1
        self.ACTION_SELL = 2
        self.SIGNAL_HOLD = 0
        self.SIGNAL_BUY = 1
        self.SIGNAL_SELL = -1

        # Telegram settings
        self.TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
config = Config()

print(f"✅ Config loaded:")
print(f"   Exchange: BYBIT | Market: FUTURES | Leverage: {config.LEVERAGE}x")
print(f"   Position Size: {config.MAX_POSITION_SIZE_PERCENT}% | Min Trade: ${config.MIN_TRADE_AMOUNT_USDT}")
print(f"   Mode: {'DRY RUN' if config.DRY_RUN else 'LIVE'}")
print(f"   Trading top {config.TOP_COINS_COUNT} volume coins (scanner ON)")
print(f"   Excluded symbols: {config.EXCLUDED_SYMBOLS}")
print(f"   TP/SL: {'DYNAMIC (ATR)' if config.USE_TP_SL else 'OFF'}")
print(f"   Trailing Stop: {'ON' if config.USE_TRAILING_STOP else 'OFF'}")