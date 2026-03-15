import asyncio
import ray
from distributed_trainer import DistributedTrainer
from loguru import logger

async def main():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_gpus=4)
        logger.info("Ray initialized")

    trainer = DistributedTrainer()
    future = trainer.start("temp_training_data.pkl", "models/ppo_trader.zip", "models/ppo_trader_new.zip")
    if future:
        result = await future
        logger.info(f"Training result: {result}")
    else:
        logger.error("Failed to start training")
    ray.shutdown()

if __name__ == "__main__":
    asyncio.run(main())