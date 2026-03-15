import asyncio
import pandas as pd
from loguru import logger
from rabbitmq_client import RabbitMQClient, RABBITMQ_URL
from retest_detector import RetestDetector, RetestSignal
from config import config

class RetestAgent:
    def __init__(self):
        self.detector = RetestDetector()
        self.name = "Retest Scanner"

    def analyze(self, df: pd.DataFrame) -> dict:
        signal = self.detector.detect_retest(df)
        if signal is None:
            return {
                'signal': config.SIGNAL_HOLD,
                'confidence': 0.0,
                'reasoning': "No retest pattern detected",
                'data': {}
            }
        if signal.direction == 'long':
            sig = config.SIGNAL_BUY
        else:
            sig = config.SIGNAL_SELL
        return {
            'signal': sig,
            'confidence': signal.confidence,
            'reasoning': signal.reason,
            'data': {
                'breakout_price': signal.breakout_price,
                'retest_price': signal.retest_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit
            }
        }

_agent = RetestAgent()

async def process_task(task: dict) -> dict:
    df_dict = task['df']
    df = pd.DataFrame(df_dict)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    result = _agent.analyze(df)
    return result

async def main():
    client = RabbitMQClient(RABBITMQ_URL)
    await client.connect()
    logger.info("Retest worker started...")
    await client.start_worker("retest_queue", process_task)

if __name__ == "__main__":
    asyncio.run(main())