import asyncio
import nest_asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from binanceclient import BinanceClient
import time

nest_asyncio.apply()

binance_client = BinanceClient()

chat_states = {}

async def send_binance_balance(chat_id, bot):
    while True:
        balance = binance_client.get_balance()
        await bot.send_message(chat_id, f"Current Binance Balance: {balance}")
        await asyncio.sleep(10)

# async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     chat_id = update.effective_chat.id
#     asyncio.create_task(send_binance_balance(chat_id, context.bot))
#     await update.message.reply_text(f'Balance updates started.')

async def main():
    app = ApplicationBuilder().token("6520299940:AAFy4rkar4ZJNUpHkE1TlgOBv4iUGYhkys8").build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("set_drawdown", set_drawdown))

    await app.run_polling()

def calculate_drawdown(initial_balance, current_balance):
    initial_balance = float(initial_balance)
    current_balance = float(current_balance)
    return ((initial_balance - current_balance) / initial_balance) * 100


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    initial_balance = binance_client.get_balance()

    # Initialize or reset the state for this chat
    chat_states[chat_id] = {
        "initial_balance": initial_balance,
        "max_drawdown": 3.0,
        "monitoring": True
    }
    asyncio.create_task(monitor_drawdown(chat_id, context.bot))
    await update.message.reply_text(f'Monitoring for drawdown started with initial balance: {initial_balance}')

async def monitor_drawdown(chat_id, bot):
    status_update_interval = 1
    last_status_update = time.time()

    while chat_id in chat_states and chat_states[chat_id]["monitoring"]:
        current_balance = binance_client.get_balance()
        initial_balance = chat_states[chat_id]["initial_balance"]
        max_drawdown = chat_states[chat_id]["max_drawdown"]
        drawdown = calculate_drawdown(initial_balance, current_balance)

        if drawdown >= max_drawdown:
            await bot.send_message(chat_id, f"Alert: Maximum drawdown of {max_drawdown}% reached. Current drawdown: {drawdown:.2f}%")
            initial_balance = current_balance

        if time.time() - last_status_update > status_update_interval:
            await bot.send_message(chat_id, "Monitoring is active.")
            last_status_update = time.time()

        await asyncio.sleep(1)


async def set_drawdown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    try:
        max_drawdown = float(context.args[0])

        if chat_id in chat_states:
            chat_states[chat_id]["monitoring"] = False
            await asyncio.sleep(1)

        chat_states[chat_id] = {
            "initial_balance": binance_client.get_balance(),
            "max_drawdown": max_drawdown,
            "monitoring": True
        }
        asyncio.create_task(monitor_drawdown(chat_id, context.bot))
        await update.message.reply_text(f'Max drawdown set to {max_drawdown}%. Monitoring restarted.')
    except (IndexError, ValueError):
        await update.message.reply_text('Usage: /set_drawdown <percentage>')


if __name__ == '__main__':
    asyncio.run(main())
