from decouple import config
from agents import AsyncOpenAI, OpenAIChatCompletionsModel , Agent, Runner, set_tracing_disabled
set_tracing_disabled(True)

key = config("GEMINI_API_KEY")
base_url = config("BASE_URL")

gemini_client = AsyncOpenAI(api_key=key, base_url=base_url)

MODEL = OpenAIChatCompletionsModel("gemini-2.5-flash", openai_client=gemini_client)

agent = Agent(name="Yousra", instructions="You are a helpful assistant.Give me answer in detail.", model=MODEL)

res = Runner.run_sync(starting_agent=agent, input="4+4=?")
print(res.final_output)

# print(key)
# print(base_url)