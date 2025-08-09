# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 21:39:33 2025

@author: BlankAdventure
"""

import os
import random
import logging
import asyncio
from collections import deque
from typing import Deque
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
import google.generativeai as genai
from google.generativeai.types import GenerationConfig



# === CONFIGURATION ===
TELEGRAM_BOT_TOKEN = os.environ.get("telegram_bot")
GEMINI_API_KEY = os.environ.get("genai_key")

TEMPERATURE = 1.1
MAX_HISTORY = 6
MAX_CALLS = 50
MAX_WORD_LENGTH = 12
THROTTLE_INTERVAL = 7  # seconds

SYSTEM_INSTRUCTION = """
You are in a hash house harriers chat group. You like sending creative, dirty acronyms inspired by the conversation.

- The acronym words must form a proper sentence.
- THe sentence should relate to the conversation if possible.
- Use only alphabetic characters.
- Reply with only the sentence.
"""

PROMPT_TEMPLATE = """
# CONVERSATION
{convo}

Now generate an acronym for the word "{word}".
"""

# === SETUP ===
generation_config = GenerationConfig(temperature=TEMPERATURE)
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash", system_instruction=SYSTEM_INSTRUCTION)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BotState:
    def __init__(self):
        self.event_queue: Deque[tuple[any,any]] = deque()
        self.queue_event: asyncio.Event = asyncio.Event()
        self.history: list = []
        self.call_count: int = 0

state = BotState()

# === QUEUE PROCESSOR ===
async def queue_processor():
    while True:
        if not state.event_queue:
            state.queue_event.clear()
            await state.queue_event.wait()

        if state.event_queue:
            update, prompt = state.event_queue.popleft()
            try:
                response = await asyncio.to_thread(model.generate_content, 
                                                   prompt,
                                                   generation_config=generation_config)                
                await update.message.reply_text(response.text.strip())
            except Exception as e:
                logger.error(f"Model error: {e}")
                await update.message.reply_text("Dammit you broke something")

            await asyncio.sleep(THROTTLE_INTERVAL)


# === COMMAND HANDLERS ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi, I'm Acrobot. Use /acro WORD to generate an acronym.")

async def info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Chat History:\n" + "\n".join(f"{u}: {m}" for u, m in state.history))
    logger.info(f"Queue length: {len(state.event_queue)} | API calls: {state.call_count}")
    await update.message.reply_text(
        f"Queue length: {len(state.event_queue)} | API calls: {state.call_count}"
    )


async def add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 2:
        await update.message.reply_text("Usage: /add username message")
        return

    username, message = context.args[0], " ".join(context.args[1:])
    state.history.append((username, message))
    state.history = state.history[-MAX_HISTORY:]
    await update.message.reply_text("Message added.")


# === MESSAGE HANDLER ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    sender = user.username or user.first_name or user.last_name or "Unknown"
    message = update.message.text

    state.history.append((sender, message))
    state.history = state.history[-MAX_HISTORY:]


# === ACRONYM GENERATOR ===
async def generate_acronym(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if state.call_count >= MAX_CALLS:
        await update.message.reply_text("No more! You're wasting my precious tokens!")
        return

    word = context.args[0] if context.args else random.choice(
        state.history[-1][1].split()
    )
    word = word[:MAX_WORD_LENGTH]

    convo = "\n".join(f"{u}: {m}" for u, m in state.history)
    prompt = PROMPT_TEMPLATE.format(convo=convo, word=word)

    state.event_queue.append((update, prompt))
    state.queue_event.set()
    state.call_count += 1


# === MAIN FUNCTION ===
def main():
    loop = asyncio.get_event_loop()
    loop.create_task(queue_processor())

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("info", info))    
    app.add_handler(CommandHandler("add", add))
    app.add_handler(CommandHandler("acro", generate_acronym))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
