import asyncio
import json
import uuid
from typing import Optional, Dict, Any
from loguru import logger
import aio_pika

RABBITMQ_URL = "amqp://guest:guest@localhost/"

class RabbitMQClient:
    def __init__(self, amqp_url: str = RABBITMQ_URL):
        self.amqp_url = amqp_url
        self.connection = None
        self.channel = None

    async def connect(self):
        self.connection = await aio_pika.connect_robust(self.amqp_url)
        self.channel = await self.connection.channel()
        logger.info("✅ Connected to RabbitMQ")

    async def close(self):
        if self.connection:
            await self.connection.close()
            logger.info("🔌 RabbitMQ connection closed")

    async def publish_task(self, queue_name: str, task_data: Dict[str, Any], timeout: int = 30) -> Optional[Dict]:
        callback_queue = await self.channel.declare_queue(exclusive=True)
        correlation_id = str(uuid.uuid4())

        await self.channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(task_data).encode(),
                correlation_id=correlation_id,
                reply_to=callback_queue.name,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=queue_name,
        )

        logger.debug(f"Waiting for reply from {queue_name} (corr_id={correlation_id}) with timeout {timeout}s")
        try:
            result = await asyncio.wait_for(
                self._wait_for_reply(callback_queue, correlation_id),
                timeout=timeout
            )
            logger.debug(f"Reply received from {queue_name}")
            return result
        except asyncio.TimeoutError:
            logger.error(f"⏰ Timeout waiting for response from {queue_name} (correlation_id={correlation_id})")
            return None

    async def _wait_for_reply(self, callback_queue, correlation_id):
        async with callback_queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    if message.correlation_id == correlation_id:
                        return json.loads(message.body.decode())

    async def start_worker(self, queue_name: str, callback):
        await self.channel.set_qos(prefetch_count=1)
        queue = await self.channel.declare_queue(queue_name, durable=True)

        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    try:
                        body = json.loads(message.body.decode())
                        body['correlation_id'] = message.correlation_id
                        logger.debug(f"📥 Received task from {queue_name} (correlation_id={message.correlation_id})")
                        result = await callback(body)
                        await self.channel.default_exchange.publish(
                            aio_pika.Message(
                                body=json.dumps(result).encode(),
                                correlation_id=message.correlation_id,
                            ),
                            routing_key=message.reply_to,
                        )
                        logger.debug(f"📤 Sent result for {message.correlation_id}")
                    except Exception as e:
                        logger.error(f"❌ Worker error: {e}")