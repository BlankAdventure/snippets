# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 13:45:53 2025

@author: BlankAdventure
"""

import acrobot
import uvicorn
from telegram import Update
from http import HTTPStatus
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-p', help='server port (listening)',type=int)
parser.add_argument('-a', help='server IP address (listening)',type=str)
parser.add_argument('-w', help='webhook URL', default=None,type=str)
args = parser.parse_args()
webhook_url =  args.w 

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    app.state.bot_app = acrobot.bot_builder()
    if webhook_url: 
        await app.state.bot_app.bot.setWebhook(webhook_url)
    async with app.state.bot_app:
        await app.state.bot_app.start()
        yield
        await app.state.bot_app.stop()    

# Initialize the FastAPI application with the lifespan handler.
app = FastAPI(lifespan=lifespan)

# This is the endpoint that Telegram will hit with updates.
# We're listening for POST requests on the path we defined.
@app.post('/')
async def webhook_handler(request: Request):
    """Processes incoming Telegram updates from the webhook."""
    json_string = await request.json()
    update = Update.de_json(json_string, app.state.bot_app.bot)
    await app.state.bot_app.process_update(update)
    return Response(status_code=HTTPStatus.OK)


uvicorn.run(app, host=args.a, port=args.p)
    
