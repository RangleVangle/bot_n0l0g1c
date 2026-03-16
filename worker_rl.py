import asyncio
import pandas as pd
import numpy as np
import os
import glob
from loguru import logger
from rabbitmq_client import RabbitMQClient, RABBITMQ_URL
import ray
import torch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
import gymnasium as gym
from gymnasium import spaces


# ===== Регистрация среды (необходимо для загрузки алгоритма) =====
class DummyTradingEnv(gym.Env):
    """Заглушка среды для загрузки алгоритма Ray."""
    def __init__(self, config=None):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(363,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
    def reset(self, seed=None, options=None):
        return self.observation_space.sample(), {}
    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}

def env_creator(env_config):
    return DummyTradingEnv(env_config)

register_env("TradingEnv", env_creator)
# ================================================================

CHECKPOINT_DIR = "models/rllib_checkpoints"
if not os.path.isdir(CHECKPOINT_DIR):
    raise FileNotFoundError(f"Папка с чекпоинтами {CHECKPOINT_DIR} не найдена.")

# Ищем самый свежий чекпоинт
checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_*"))
if checkpoints:
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    logger.info(f"Загружаем модель Ray из {latest_checkpoint}")
    checkpoint_path = latest_checkpoint
else:
    checkpoint_path = CHECKPOINT_DIR

checkpoint_path = os.path.abspath(checkpoint_path)
logger.info(f"Абсолютный путь к чекпоинту: {checkpoint_path}")

# Инициализируем Ray с одной GPU
ray.init(ignore_reinit_error=True, num_gpus=1)
algo = Algorithm.from_checkpoint(checkpoint_path)
module = algo.get_module()

# Определяем устройство: cuda, если доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Используем устройство: {device}")
module = module.to(device)

# Класс для подготовки фич (как раньше)
class FeaturePreparer:
    def __init__(self):
        self.feature_columns = ['close', 'volume', 'rsi', 'macd', 'bb_position', 'volatility']
        self.window_size = 60

    def prepare(self, df: pd.DataFrame) -> np.ndarray:
        from strategies import TechnicalIndicatorCalculator
        calc = TechnicalIndicatorCalculator()
        df_features = calc.calculate_all(df.tail(100))
        if len(df_features) < self.window_size:
            base = np.zeros(self.window_size * len(self.feature_columns))
        else:
            window = df_features.iloc[-self.window_size:][self.feature_columns].values
            base = window.flatten().astype(np.float32)
        macro = np.zeros(3, dtype=np.float32)
        return np.concatenate([base, macro])

feature_preparer = FeaturePreparer()

async def process_task(task: dict) -> dict:
    try:
        df_dict = task['df']
        df = pd.DataFrame(df_dict)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

        obs = feature_preparer.prepare(df)
        obs_tensor = torch.from_numpy(obs).float().to(device)
        obs_tensor = obs_tensor.unsqueeze(0)
        input_dict = {"obs": obs_tensor}

        with torch.no_grad():
            out = module.forward_inference(input_dict)

        logits = out["action_dist_inputs"]
        action = int(logits.argmax(dim=-1).cpu().numpy()[0])

        if action == 1:
            signal = 1
            confidence = 0.8
            reasoning = "RL suggests BUY (Ray model on GPU)"
        elif action == 2:
            signal = -1
            confidence = 0.8
            reasoning = "RL suggests SELL (Ray model on GPU)"
        else:
            signal = 0
            confidence = 0.5
            reasoning = "RL suggests HOLD (Ray model on GPU)"

        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': reasoning,
            'data': {'action': int(action)}
        }
    except Exception as e:
        logger.error(f"Ошибка при обработке задачи: {e}")
        return {
            'signal': 0,
            'confidence': 0.5,
            'reasoning': f"RL agent error: {e}",
            'data': {}
        }

async def main():
    client = RabbitMQClient(RABBITMQ_URL)
    await client.connect()
    logger.info("RL worker (Ray on GPU) started, waiting for tasks...")
    await client.start_worker("rl_queue", process_task)

if __name__ == "__main__":
    asyncio.run(main())