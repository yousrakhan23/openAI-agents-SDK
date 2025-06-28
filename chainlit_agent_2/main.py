from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig
import chainlit as cl
from dotenv import load_dotenv
import os
import datetime

# Load environment variables
load_dotenv()

# Set up external client
gemini_api_key = os.getenv("GEMINI_API_KEY")
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Define model and config
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=False,  # Enable tracing for better debugging/logging in real-world usage
)

# Create a more functional agent
agent = Agent(
    name="Welcoming Agent",
    instructions="""
You are a friendly AI assistant created by Yousra Khan.
Greet users warmly, then offer intelligent help across various domains such as tech, business, lifestyle, coding, etc.
Keep answers helpful and conversational. Ask clarifying questions when necessary.
If the user seems idle or confused, gently offer suggestions.
""".strip(),
)

# Utility: Create session logs
def log_interaction(user_input, response):
    log_path = "chat_logs/session_log.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n[{timestamp}] User: {user_input}\n[{timestamp}] Assistant: {response}\n")

# On chat start
@cl.on_chat_start
async def handle_start():
    cl.user_session.set("history", [])
    welcome_message = (
        "👋 Hello! I am your assistant powered by Gemini.\n\n"
        "🤝Ask me anything — from coding questions to daily advice.\n\n"
        "🤞Tip: Type 'help' to see what I can do."
    )
    await cl.Message(content=welcome_message).send()

# On each user message
@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")
    user_input = message.content.strip().lower()
    history.append({"role": "user", "content": message.content})

    # Custom response for "Who is Yousra Khan?"
    if any(q in user_input for q in ["who is yousra khan", "your creator", "who made you", "who created you"]):
        response = (
            "**Yousra Khan** is the creator of this assistant. 💻\n\n"
            "She is a passionate and skilled **Web Developer** with a flair for creative design, "
            "modern frontend technologies, and building user-friendly digital experiences.\n\n"
            "You're chatting with one of her projects right now. 😊"
            "**🔗 Connect With Yousra:**\n"
            "- 🌐 [Portfolio](https://portfolio-yousra.vercel.app/)\n"
            "- 📱 [Behance](https://www.behance.net/yousrakhan7)\n"
            "- 💼 [LinkedIn](https://www.linkedin.com/in/hafiza-yousra-khan-/)\n"
            "- 🐦 [Twitter / X](https://x.com/iyousrakhan?t=szVPizq5XmGfMdKY7JABDw&s=08 )\n"
            "- 🤍 [Github](https://github.com/yousrakhan23)\n"
            "- 📷 [Instagram](https://www.instagram.com/_fumodoarishika?utm_source=qr&igsh=MWdpazkyOGVwdnd2MA==)\n"
            "- 📷 [facebook](https://www.facebook.com/share/1CNyo8vYBv/)\n"
            
            "\nFeel free to explore her work!"
        )
        await cl.Message(content=response).send()
        return

    # Help command
    if user_input == "help":
        help_text = (
           "Here are a few things I can help you with:\n"
            "- ✅ Programming help (Python, JS, etc.)\n"
            "- 📈 Business/Startup advice\n"
            "- 🌍 General knowledge & research\n"
            "- 💬 Language improvement or translation\n"
            "- 🧠 Creative writing & blog ideas\n"
            "- 🤖 AI/ML concepts and tools\n"
            "- 📊 Math & statistics\n"
            "- 🏠 Lifestyle tips (health, travel, etc.)\n\n"
            "Just type your question or topic, and I'll do my best to assist you!\n"
        )
        await cl.Message(content=help_text).send()
        return

    # Default LLM response
    result = await Runner.run(
        agent,
        input=history,
        run_config=config
    )
    response = result.final_output
    history.append({"role": "assistant", "content": response})
    cl.user_session.set("history", history)

    # Optional: Save to log
    log_interaction(message.content, response)

    await cl.Message(content=response).send()
