from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

chat_model = ChatOpenAI(
    model="meta-llama/llama-3-8b-instruct",  # ✅ replace with valid OpenRouter model
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# Chat history dictionary
chat_history = {"messages": []}

# Add system message (sets behavior of AI)
system_message = SystemMessage(content="You are a helpful AI assistant. Keep responses short and clear.")

print("🤖 Chatbot is ready! (type 'exit' to quit)")
print("------------------------------------------------")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Build message objects (system + history)
    message_objects = [system_message]
    for turn in chat_history["messages"]:
        message_objects.append(HumanMessage(content=turn["human"]))
        message_objects.append(AIMessage(content=turn["ai"]))

    # Add new user input
    message_objects.append(HumanMessage(content=user_input))

    # Get AI response
    result = chat_model.invoke(message_objects)

    # Save turn in dictionary
    chat_history["messages"].append({
        "human": user_input,
        "ai": result.content
    })

    print("AI:", result.content)

#print("\nFinal conversation history:", chat_history)
