# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 19:54:58 2025

@author: BlankAdventure
"""
import asyncio


from collections import deque
import os
import random

import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import google.generativeai as genai

TELEGRAM_BOT_TOKEN = os.environ.get("telegram_bot")
GEMINI_API_KEY = os.environ.get("genai_key")

system_instruction = """
You are a hash house harrier in a chat group who likes sending creative, dirty acronyms
inspired by the conversation."""

template = """
{convo}

Now generate an acronym for the word "{word}". 
- The acronym should form a proper sentence.
- It should relate to the chat if possible.
- Use only alphabetic characters.
- Reply with only the acronym.
"""

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash',system_instruction=system_instruction)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


history = []
n_msg = 5
max_calls = 50
num_calls = 0

# === CONFIG ===
THROTTLE_INTERVAL = 7  # seconds between processing events

# === QUEUE SETUP ===
event_queue = deque()
queue_event = asyncio.Event()

async def queue_processor():
    while True:
        if not event_queue:
            queue_event.clear()
            await queue_event.wait()
        if event_queue:            
            update, prompt = event_queue.popleft()            
            try:
                response = await call_model_async(prompt)            
                acronym = response.text.strip()
                await update.message.reply_text(acronym)
            except Exception as e:
                logger.error(f"error: {e}")
                await update.message.reply_text("Dammit you broke something")                
            await asyncio.sleep(THROTTLE_INTERVAL)  # throttle interval

async def queue(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print (f'Q-len: {len(event_queue)} | num_calls: {num_calls}')

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi, I'm an acrobot. A for an acronym by typing: /acro WORD")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global history

    user = update.message.from_user
    username = user.username
    first_name = user.first_name
    last_name = user.last_name

    message = update.message.text

    history.append( ( username or first_name or last_name or "", message)  )
    history = history[-n_msg:]    

    
async def show(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(history)

async def add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args    
    text = " ".join(context.args[1:])    
    history.append( (args[0],text) )    

async def call_model_async(prompt):
    return await asyncio.to_thread(model.generate_content, prompt)
    #def blocking_function():
    #    result = model.generate_content(prompt)
    #    return result
    #result = await asyncio.to_thread(blocking_function)
    #return result

#async def process_async(data):
#    return await asyncio.to_thread(long_running, data)


async def generate_acronym(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global num_calls
    
    args = context.args
    if args:
        word = args[0]
    else:
        word = random.choice( history[-1][1].split(" ") ) 
    
    convo = ''
    for entry in history:
        convo += entry[0] + ": " + entry[1] + "\n"
        
    prompt = template.format(convo=convo, word=word) 

    if num_calls < max_calls:
        event_queue.append((update, prompt))
        queue_event.set()  # signal the queue processor
        num_calls += 1
    else:
        await update.message.reply_text("No more! You're wasting my precious tokens!")

def main():
    loop = asyncio.get_event_loop()
    loop.create_task(queue_processor())
    
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("queue", queue))
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("acro", generate_acronym))
    app.add_handler(CommandHandler("show", show))
    app.add_handler(CommandHandler("add", add))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot is running...")
    app.run_polling()

if __name__ == '__main__':
    main()




#     prompt = f"""
# You are a creative acronym generator.

# The user previously said: {memory}

# Now, generate a creative acronym for the word "{word}". 
# Make sure it makes sense and connects with previous messages if possible.
# """