#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ccxt
import pandas as pd
import time
import os
import sys
from datetime import datetime, timedelta
import certifi
import logging
from typing import Optional

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
#  НАСТРОЙКИ
# ------------------------------------------------------------------
DEFAULT_TIMEFRAME = '5m'
DEFAULT_YEARS = 3
MAX_RETRIES = 5
RETRY_DELAY = 5
# ------------------------------------------------------------------

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
    'ARB/USDT', 'OP/USDT', 'LDO/USDT', 'RNDR/USDT',
    'FET/USDT', 'AGIX/USDT', 'OCEAN/USDT', 'APT/USDT',
    'SUI/USDT', 'SEI/USDT', 'TIA/USDT', 'INJ/USDT',
    'BLUR/USDT', 'PEPE/USDT', 'WLD/USDT', 'PYTH/USDT',
    'JUP/USDT', 'JTO/USDT', 'BONK/USDT', 'ORDI/USDT',
    'SATS/USDT', 'RATS/USDT', 'BONK/USDT',
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

def get_last_timestamp_from_csv(csv_path: str) -> Optional[int]:
    """Читает последнюю строку CSV и возвращает timestamp в миллисекундах."""
    if not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2:  # заголовок + хотя бы одна строка данных
                return None
            last_line = lines[-1].strip()
            if not last_line:
                return None
            # Первый элемент — timestamp (предполагаем, что колонки: timestamp, open, high, low, close, volume)
            ts_str = last_line.split(',')[0].strip().strip('"')
            # Пробуем преобразовать в число (миллисекунды)
            try:
                ts = int(float(ts_str))
                # Если число большое (>1e10), это миллисекунды
                if ts > 1_000_000_000_0:  # больше 10 миллиардов -> миллисекунды
                    return ts
                else:
                    # Возможно, это секунды, тогда умножаем на 1000
                    return ts * 1000
            except ValueError:
                # Если не число, пробуем распарсить как дату
                ts = pd.to_datetime(ts_str).timestamp() * 1000
                return int(ts)
    except Exception as e:
        logger.warning(f"Не удалось прочитать последнюю строку {csv_path}: {e}. Будет выполнена полная перезагрузка.")
        # Удаляем повреждённый файл
        try:
            os.remove(csv_path)
            logger.info(f"Повреждённый файл {csv_path} удалён.")
        except:
            pass
        return None

def download_with_retry(exchange, symbol, timeframe, since, limit):
    """Загружает свечи с повторными попытками при ошибках."""
    for attempt in range(MAX_RETRIES):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        except ccxt.AuthenticationError as e:
            logger.error(f"Ошибка аутентификации: {e}. Проверьте API ключи.")
            raise
        except ccxt.BadSymbol as e:
            logger.warning(f"Символ {symbol} не найден на бирже. Пропускаем.")
            return None
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"Ошибка при загрузке {symbol} (попытка {attempt+1}/{MAX_RETRIES}): {e}. Повтор через {RETRY_DELAY}с...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"Не удалось загрузить {symbol} после {MAX_RETRIES} попыток: {e}")
                raise
    return None

def download_symbol_incremental(exchange, symbol, timeframe, years, base_dir='data'):
    """Загружает только новые данные для символа, с автоматическим восстановлением."""
    tf_dir = os.path.join(base_dir, timeframe)
    os.makedirs(tf_dir, exist_ok=True)

    safe = symbol.replace('/', '_')
    csv_file = os.path.join(tf_dir, f"{safe}_{timeframe}.csv")

    end_ts = int(time.time() * 1000)
    start_ts = end_ts - int(years * 365 * 24 * 60 * 60 * 1000)

    last_ts = get_last_timestamp_from_csv(csv_file)
    if last_ts:
        # Если последняя свеча слишком старая (более 1 дня), начинаем с неё + 1 мс
        if end_ts - last_ts < 24 * 3600 * 1000:
            logger.info(f"⏭️  {symbol} ({timeframe}) уже актуален (последняя свеча: {datetime.fromtimestamp(last_ts/1000).strftime('%Y-%m-%d %H:%M')})")
            return True
        since = last_ts + 1
        logger.info(f"📥 Докачка {symbol} ({timeframe}) с {datetime.fromtimestamp(since/1000)}")
    else:
        since = start_ts
        logger.info(f"📥 Полная загрузка {symbol} ({timeframe}) за {years} лет")

    all_data = []
    limit = 200
    last_progress = 0

    try:
        while since < end_ts:
            ohlcv = download_with_retry(exchange, symbol, timeframe, since, limit)
            if ohlcv is None:
                return False
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1

            if len(all_data) > last_progress + 5000:
                last_progress = len(all_data)
                logger.info(f"  {symbol}: {len(all_data)} свечей скачано")
            time.sleep(0.5)

        if not all_data:
            logger.info(f"  ⚠️ Нет новых данных для {symbol}")
            return True

        # Если файл уже существовал, читаем старые данные
        if os.path.exists(csv_file) and last_ts:
            try:
                old_df = pd.read_csv(csv_file)
                if 'timestamp' not in old_df.columns:
                    raise ValueError("CSV не содержит колонку timestamp")
                old_df['timestamp'] = pd.to_datetime(old_df['timestamp'])
            except Exception as e:
                logger.warning(f"Не удалось прочитать старый CSV {csv_file}: {e}. Начинаем заново.")
                old_df = pd.DataFrame()
        else:
            old_df = pd.DataFrame()

        new_df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')

        if not old_df.empty:
            combined = pd.concat([old_df, new_df]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            combined.to_csv(csv_file, index=False)
            logger.info(f"  💾 Обновлён {csv_file}: добавлено {len(new_df)} новых свечей, всего {len(combined)}")
        else:
            new_df.to_csv(csv_file, index=False)
            logger.info(f"  💾 Создан {csv_file}: {len(new_df)} свечей")

        # Пытаемся сохранить Parquet
        try:
            parquet_file = csv_file.replace('.csv', '.parquet')
            df_all = pd.read_csv(csv_file)
            df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
            df_all.set_index('timestamp', inplace=True)
            df_all.to_parquet(parquet_file)
            logger.info(f"  📦 Parquet обновлён")
        except Exception as e:
            logger.debug(f"Parquet не сохранён: {e}")

        return True

    except Exception as e:
        logger.error(f"  ❌ Критическая ошибка для {symbol}: {e}")
        return False

def main():
    logger.info("🚀 ИНКРЕМЕНТАЛЬНАЯ ЗАГРУЗКА ДАННЫХ С BYBIT (улучшенная версия)")
    logger.info("=" * 60)

    timeframe, years = parse_args()
    logger.info(f"⚙️  Таймфрейм: {timeframe}")
    logger.info(f"⚙️  Глубина (для новых файлов): {years} лет")
    logger.info("📁 Данные будут докачаны/обновлены в папке 'data/<таймфрейм>/'")

    # Используем certifi для корректного пути к сертификатам
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

    exchange = ccxt.bybit({
        'enableRateLimit': True,
        'options': {'defaultType': 'linear'}
    })

    pairs_to_download = TARGET_SYMBOLS
    logger.info(f"🎯 Будет обработано {len(pairs_to_download)} монет")

    successful = []
    failed = []

    for i, sym in enumerate(pairs_to_download, 1):
        logger.info(f"\n📊 [{i}/{len(pairs_to_download)}] {sym}")
        if download_symbol_incremental(exchange, sym, timeframe, years):
            successful.append(sym)
        else:
            failed.append(sym)
        time.sleep(2)

    logger.info("\n" + "=" * 60)
    logger.info("📈 **ИТОГ**")
    logger.info(f"✅ Успешно: {len(successful)}")
    logger.info(f"❌ Ошибки: {len(failed)}")
    if failed:
        logger.info(f"Проблемные монеты: {', '.join(failed)}")

if __name__ == "__main__":
    main()