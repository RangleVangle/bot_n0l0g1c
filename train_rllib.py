import os
import argparse
import pickle
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

# ------------------------------------------------------------------
#  Среда для воспроизведения сохранённых опытов
# ------------------------------------------------------------------
class ReplayTradingEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.df = config['df']  # DataFrame с колонками: observation, action, reward
        self.current_idx = 0
        # Наблюдение – вектор из 363 чисел (60*6 + 3 макро)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(363,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        self.current_idx = 0
        obs = self.df.iloc[self.current_idx]['observation']
        if isinstance(obs, list):
            obs = np.array(obs, dtype=np.float32)
        return obs, {}

    def step(self, action):
        exp = self.df.iloc[self.current_idx]
        # Истинное действие (не используется напрямую, но может пригодиться)
        true_action = exp['action']
        reward = exp['reward']
        done = self.current_idx >= len(self.df) - 1
        self.current_idx += 1
        if not done:
            next_obs = self.df.iloc[self.current_idx]['observation']
            if isinstance(next_obs, list):
                next_obs = np.array(next_obs, dtype=np.float32)
        else:
            next_obs = np.zeros(363, dtype=np.float32)
        return next_obs, reward, done, False, {}

# ------------------------------------------------------------------
#  Основная функция обучения
# ------------------------------------------------------------------
def train_rllib(data_path, checkpoint_dir, total_timesteps):
    # Загружаем данные
    df = pd.read_pickle(data_path)
    if df.empty:
        print("❌ Empty dataset, exiting")
        return False

    # Регистрируем среду
    def env_creator(env_config):
        return ReplayTradingEnv(env_config)
    register_env("ReplayTradingEnv", env_creator)

    # Конфигурация PPO
    config = (
        PPOConfig()
        .environment("ReplayTradingEnv", env_config={"df": df})
        .training(
            gamma=0.99,
            lr=3e-4,
            train_batch_size=4000,
            sgd_minibatch_size=512,
            num_sgd_iter=10,
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "tanh",
            },
        )
        .resources(num_gpus=2)  # используем 2 GPU
        .rollouts(num_rollout_workers=0)  # без воркеров, т.к. данные фиксированы
    )

    # Пытаемся загрузить существующий чекпоинт
    algo = None
    if os.path.exists(checkpoint_dir):
        try:
            algo = Algorithm.from_checkpoint(checkpoint_dir)
            print(f"🔄 Loaded existing checkpoint from {checkpoint_dir}")
        except Exception as e:
            print(f"⚠️ Could not load checkpoint: {e}. Creating new algorithm.")
            algo = config.build()
    else:
        print("🆕 Creating new algorithm")
        algo = config.build()

    # Обучаем, пока не достигнем нужного числа шагов
    total = 0
    while total < total_timesteps:
        result = algo.train()
        total = result["timesteps_total"]
        print(f"   timesteps: {total}, reward: {result.get('episode_reward_mean', 0):.2f}")

    # Сохраняем чекпоинт
    save_path = algo.save(checkpoint_dir)
    print(f"✅ Model saved to {save_path}")
    return True

# ------------------------------------------------------------------
#  Точка входа
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to pickle file with experiences")
    parser.add_argument("--checkpoint_dir", default="models/rllib_checkpoints", help="Directory to save/load checkpoints")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total timesteps to train")
    args = parser.parse_args()

    # Инициализируем Ray (если ещё не инициализирован)
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    success = train_rllib(args.data, args.checkpoint_dir, args.timesteps)
    ray.shutdown()
    exit(0 if success else 1)