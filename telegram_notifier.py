import asyncio
import aiohttp
from loguru import logger

class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.session = None

    async def _get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    async def send_message(self, text: str, parse_mode: str = "HTML"):
        """Отправляет сообщение в Telegram."""
        session = await self._get_session()
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True
        }
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Telegram send error: {resp.status} - {error_text}")
                else:
                    logger.debug("Telegram message sent")
        except Exception as e:
            logger.error(f"Telegram exception: {e}")

    async def send_trade_notification(self, trade_data: dict):
        """Форматирует и отправляет уведомление о сделке с эмодзи."""
        pnl = trade_data.get('pnl', 0)
        if pnl > 0:
            profit_emoji = "🟢💰"  # зелёный + деньги
            result_text = f"PROFIT +${pnl:.2f}"
        elif pnl < 0:
            profit_emoji = "🔴💸"  # красный + улетающие деньги
            result_text = f"LOSS -${abs(pnl):.2f}"
        else:
            profit_emoji = "⚪"
            result_text = "BREAK EVEN"

        # Эмодзи для типа сделки
        if "BUY" in trade_data['type'].upper():
            type_emoji = "📈"
        elif "SELL" in trade_data['type'].upper():
            type_emoji = "📉"
        else:
            type_emoji = "🔄"

        text = f"""
    {profit_emoji} <b>TRADE RESULT</b>
    {type_emoji} Symbol: {trade_data['symbol']}
    Type: {trade_data['type']}
    Price: ${trade_data['price']:.2f}
    Amount: {trade_data.get('amount', 'N/A')} contracts
    Confidence: {trade_data.get('confidence', 0):.1%}
    Regime: {trade_data.get('regime', 'Unknown')}
    Reasons: {trade_data.get('reasons', 'N/A')}
    {result_text}
    """
        await self.send_message(text)
    async def send_trade_open(self, trade_data: dict):
        """Уведомление об открытии позиции (без PnL)."""
        # Эмодзи для типа сделки
        if "BUY" in trade_data['type'].upper():
            type_emoji = "🟢📈"
        elif "SELL" in trade_data['type'].upper():
            type_emoji = "🔴📉"
        else:
            type_emoji = "⚪"

        text = f"""
    🚀 <b>NEW POSITION OPENED</b>
    {type_emoji} Symbol: {trade_data['symbol']}
    Type: {trade_data['type']}
    Price: ${trade_data['price']:.2f}
    Amount: {trade_data.get('amount', 'N/A')} contracts
    Confidence: {trade_data.get('confidence', 0):.1%}
    Regime: {trade_data.get('regime', 'Unknown')}
    Reasons: {trade_data.get('reasons', 'N/A')}
    """
        await self.send_message(text)

    async def send_error_notification(self, error_msg: str):
        """Отправляет уведомление об ошибке."""
        text = f"❌ <b>ERROR</b>\n{error_msg}"
        await self.send_message(text)

    async def close(self):
        if self.session:
            await self.session.close()