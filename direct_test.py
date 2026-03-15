from rl_train_online import train_rl
import traceback

try:
    success = train_rl(
        data_path="temp_training_data.pkl",
        model_path="models/ppo_trader.zip",
        output_path="models/ppo_trader_new.zip",
        total_timesteps_multiplier=10
    )
    print(f"Result: {success}")
except Exception as e:
    print("Exception occurred:")
    traceback.print_exc()
