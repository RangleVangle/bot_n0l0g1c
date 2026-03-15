import ccxt
import pandas as pd
import time

exchange = ccxt.bybit({
    'enableRateLimit': True,
    'options': {'defaultType': 'linear'}
})

symbol = 'BTC/USDT'  # Замените на нужный символ (например, SOL/USDT)
timeframe = '1h'     # Замените на нужный таймфрейм (1m, 5m, 15m, 1h)
limit = 1000         # Максимум свечей за один запрос
all_data = []

# Задайте начальную и конечную даты (в миллисекундах)
since = exchange.parse8601('2022-01-01T00:00:00Z')
end = exchange.parse8601('2026-03-01T00:00:00Z')

while since < end:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if len(ohlcv) == 0:
            break
        all_data.extend(ohlcv)
        since = ohlcv[-1][0] + 1  # следующий timestamp
        print(f"Скачано {len(all_data)} свечей для {symbol} ({timeframe})")
        time.sleep(1)  # пауза для соблюдения rate limit
    except Exception as e:
        print(f"Ошибка: {e}")
        break

# Сохраняем в CSV (можно потом конвертировать в Parquet)
df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.to_csv(f'{symbol.replace("/","_")}_{timeframe}.csv', index=False)
print(f"Сохранено {len(df)} свечей в файл {symbol.replace('/','_')}_{timeframe}.csv")