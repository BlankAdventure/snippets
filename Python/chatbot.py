import os
from nicegui import ui, run
from google import genai
from google.genai import types


# A simple chatbot demo combining NiceGUI and Gemini

# Set your API key in environment variables
GEMINI_API_KEY = os.getenv('your_gemini_key')  


# Provide a system instruction
sys_instruct = "You are a helpful pirate."


# Initialize the Gemini model. The chats option automatically maintains history (locally)
client = genai.Client(api_key=GEMINI_API_KEY)
chat = client.chats.create(model="gemini-2.0-flash",
                           config=types.GenerateContentConfig(system_instruction=sys_instruct)
                           )


# Function to handle sending a message
async def send_message():
    user_input = input_field.value.strip()
    if user_input:                
        # chat.send_message is not awaitable, so wrap it with run.io_bound
        await run.io_bound( chat.send_message, user_input )        
        # Update the chat history UI element
        await update_chat()       
        # Clear the user input field
        input_field.value = ""        


# Function to update the chat UI
async def update_chat():
    chat_container.clear()  # Clear previous messages
    with chat_container:
        for message in chat._curated_history:
            if message.role == "user":
                ui.label(f"You: {message.parts[0].text}").classes("text-blue-600")
            else:
                ui.label(f"Bot: {message.parts[0].text}").classes("text-green-600")


# Build the UI
with ui.column().classes("w-full max-w-2xl mx-auto p-4"):
    ui.label("Gemini Chatbot").classes("text-2xl font-bold mb-4")
    
    # Chat container
    chat_container = ui.column().classes("w-full h-96 overflow-y-auto border p-4 rounded")
    
    # Input field and send button
    with ui.row().classes("w-full mt-4"):
        input_field = ui.input(placeholder="Type your message...").classes("flex-grow")
        ui.button("Send", on_click=send_message).classes("ml-2")

    # Bind Enter key to send message
    input_field.on("keydown.enter", send_message)
    
# Run the NiceGUI app
ui.run(title="Gemini Chatbot", port=8080)