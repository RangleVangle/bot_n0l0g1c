import asyncio
import pandas as pd
from loguru import logger
from rabbitmq_client import RabbitMQClient, RABBITMQ_URL
from triangle_detector import TriangleDetector, TrianglePattern
from config import config

class TriangleAgent:
    def __init__(self):
        self.detector = TriangleDetector(
            lookback=config.TRIANGLE_LOOKBACK,
            min_touches=config.TRIANGLE_MIN_TOUCHES,
            volume_window=config.TRIANGLE_VOLUME_WINDOW,
            max_deviation=config.TRIANGLE_MAX_DEVIATION
        )
        self.name = "Triangle Scanner"

    def analyze(self, df: pd.DataFrame) -> dict:
        pattern = self.detector.find_triangle(df)
        if pattern is None:
            return {
                'signal': config.SIGNAL_HOLD,
                'confidence': 0.0,
                'reasoning': "No triangle pattern detected",
                'data': {}
            }
        breakout = self.detector.get_breakout_signal(df, pattern)
        if breakout['signal'] == config.SIGNAL_HOLD:
            return {
                'signal': config.SIGNAL_HOLD,
                'confidence': pattern.confidence * 0.5,
                'reasoning': f"{pattern.type.capitalize()} triangle forming, height {pattern.height:.1f}%",
                'data': {'pattern': pattern.type, 'height': pattern.height}
            }
        return {
            'signal': breakout['signal'],
            'confidence': breakout['confidence'],
            'reasoning': breakout['reasoning'],
            'data': {
                'pattern': pattern.type,
                'breakout': breakout,
                'target': breakout['target'],
                'stop': breakout['stop']
            }
        }

_agent = TriangleAgent()

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
    logger.info("Triangle worker started...")
    await client.start_worker("triangle_queue", process_task)

if __name__ == "__main__":
    asyncio.run(main())