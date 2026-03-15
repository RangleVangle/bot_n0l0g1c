#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ccxt
import pandas as pd
import time
import os
import sys
from datetime import datetime, timedelta

# ------------------------------------------------------------------
#  НАСТРОЙКИ
# ------------------------------------------------------------------
DEFAULT_TIMEFRAME = '5m'
DEFAULT_YEARS = 3

# Список монет (можно расширять)
TARGET_SYMBOLS = [
    'XRP/USDT', 'BTC/USDT', 'ETH/USDT',
    'SOL/USDT', 'DOGE/USDT', 'ADA/USDT', 'LINK/USDT',
    'DOT/USDT', 'AVAX/USDT', 'MATIC/USDT', 'NEAR/USDT',
    'ATOM/USDT', 'ALGO/USDT', 'FTM/USDT', 'SAND/USDT',
    'MANA/USDT', 'AXS/USDT', 'APE/USDT', 'GALA/USDT',
    'ROSE/USDT', 'KSM/USDT', 'CRV/USDT', '1INCH/USDT',
    'UNI/USDT', 'AAVE/USDT', 'MKR/USDT', 'COMP/USDT',
    'SNX/USDT', 'SUSHI/USDT', 'YFI/USDT', 'BAL/USDT',
    'ZRX/USDT', 'BAT/USDT', 'ENJ/USDT', 'CHZ/USDT',
    # Добавьте новые
    'ARB/USDT', 'OP/USDT', 'LDO/USDT', 'RNDR/USDT',
    'FET/USDT', 'AGIX/USDT', 'OCEAN/USDT', 'APT/USDT',
    'SUI/USDT', 'SEI/USDT', 'TIA/USDT', 'INJ/USDT',
    'BLUR/USDT', 'PEPE/USDT', 'WLD/USDT', 'PYTH/USDT',
    'JUP/USDT', 'JTO/USDT', 'BONK/USDT', 'ORDI/USDT',
    'SATS/USDT', 'RATS/USDT', '1000PEPE/USDT', '1000BONK/USDT',
]

def parse_args():
    tf = DEFAULT_TIMEFRAME
    years = DEFAULT_YEARS
    if len(sys.argv) > 1:
        tf = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            years = float(sys.argv[2])
        except:
            pass
    return tf, years

def get_last_timestamp_from_csv(csv_path):
    """Читает последнюю строку CSV и возвращает timestamp в миллисекундах."""
    if not os.path.exists(csv_path):
        return None
    try:
        # Читаем последнюю строку файла напрямую (быстро и надёжно)
        with open(csv_path, 'rb') as f:
            f.seek(-2, os.SEEK_END)      # переходим к предпоследнему байту
            while f.read(1) != b'\n':    # ищем начало последней строки
                f.seek(-2, os.SEEK_CUR)
            last_line = f.readline().decode()
        if not last_line:
            return None
        # Предполагаем, что первый столбец - timestamp в формате ISO
        last_ts_str = last_line.split(',')[0]
        last_ts = pd.to_datetime(last_ts_str).timestamp() * 1000
        return int(last_ts)
    except Exception as e:
        print(f"⚠️  Ошибка чтения {csv_path}: {e}")
        return None

def download_symbol_incremental(exchange, symbol, timeframe, years, base_dir='data'):
    """Загружает только новые данные для символа, если файл уже существует."""
    tf_dir = os.path.join(base_dir, timeframe)
    os.makedirs(tf_dir, exist_ok=True)

    safe = symbol.replace('/', '_')
    csv_file = os.path.join(tf_dir, f"{safe}_{timeframe}.csv")

    end_ts = int(time.time() * 1000)
    start_ts = end_ts - int(years * 365 * 24 * 60 * 60 * 1000)

    # Если файл существует, определяем последнюю дату
    last_ts = get_last_timestamp_from_csv(csv_file)
    if last_ts:
        # Если последняя дата уже близка к сегодняшнему дню (меньше чем 1 день отставания), можно пропустить
        if end_ts - last_ts < 24 * 3600 * 1000:
            print(f"⏭️  {symbol} ({timeframe}) уже актуален (последняя свеча: {datetime.fromtimestamp(last_ts/1000).strftime('%Y-%m-%d %H:%M')})")
            return True
        # Начинаем с последней + 1 мс
        since = last_ts + 1
        print(f"📥 Докачка {symbol} ({timeframe}) с {datetime.fromtimestamp(since/1000)}")
    else:
        # Файла нет – качаем всё с начала
        since = start_ts
        print(f"📥 Полная загрузка {symbol} ({timeframe}) за {years} лет")

    all_data = []
    limit = 200
    last_progress = 0

    try:
        while since < end_ts:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1

            # Прогресс
            if len(all_data) > last_progress + 5000:
                last_progress = len(all_data)
                print(f"  {symbol}: {len(all_data)} свечей скачано")
            time.sleep(0.5)

        if not all_data:
            print(f"  ⚠️ Нет новых данных для {symbol}")
            return True

        # Если файл уже существовал, читаем старые данные и объединяем
        if os.path.exists(csv_file) and last_ts:
            old_df = pd.read_csv(csv_file)
            new_df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
            # Удаляем возможные дубликаты по timestamp
            combined = pd.concat([old_df, new_df]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            combined.to_csv(csv_file, index=False)
            print(f"  💾 Обновлён {csv_file}: добавлено {len(new_df)} новых свечей, всего {len(combined)}")
        else:
            # Новый файл
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.to_csv(csv_file, index=False)
            print(f"  💾 Создан {csv_file}: {len(df)} свечей")

        # Пытаемся сохранить Parquet
        try:
            parquet_file = csv_file.replace('.csv', '.parquet')
            if os.path.exists(csv_file):
                df_all = pd.read_csv(csv_file)
                df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
                df_all.set_index('timestamp', inplace=True)
                df_all.to_parquet(parquet_file)
                print(f"  📦 Parquet обновлён")
        except:
            pass

        return True

    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
        return False

def main():
    print("🚀 ИНКРЕМЕНТАЛЬНАЯ ЗАГРУЗКА ДАННЫХ С BYBIT")
    print("=" * 60)

    timeframe, years = parse_args()
    print(f"⚙️  Таймфрейм: {timeframe}")
    print(f"⚙️  Глубина (для новых файлов): {years} лет")
    print("📁 Данные будут докачаны/обновлены в папке 'data/<таймфрейм>/'")

    exchange = ccxt.bybit({
        'enableRateLimit': True,
        'options': {'defaultType': 'linear'}
    })

    # Можно использовать все доступные пары, но для простоты используем TARGET_SYMBOLS
    pairs_to_download = TARGET_SYMBOLS
    print(f"🎯 Будет обработано {len(pairs_to_download)} монет")

    successful = []
    failed = []

    for i, sym in enumerate(pairs_to_download, 1):
        print(f"\n📊 [{i}/{len(pairs_to_download)}] {sym}")
        if download_symbol_incremental(exchange, sym, timeframe, years):
            successful.append(sym)
        else:
            failed.append(sym)
        time.sleep(2)

    print("\n" + "=" * 60)
    print("📈 **ИТОГ**")
    print(f"✅ Успешно: {len(successful)}")
    print(f"❌ Ошибки: {len(failed)}")
    if failed:
        print(f"Проблемные монеты: {', '.join(failed)}")

if __name__ == "__main__":
    main()