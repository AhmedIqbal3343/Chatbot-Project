from flask import Flask, request, jsonify, render_template  # ADD render_template
from flask_cors import CORS
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize the chat model
chat_model = ChatOpenAI(
    model="meta-llama/llama-3-8b-instruct",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

chat_history = {"messages": []}
system_message = SystemMessage(content="You are a helpful AI assistant. Keep responses short and clear.")

# Change this route to serve your HTML interface
@app.route('/')
def home():
    return render_template('index.html')  # This serves your HTML file

@app.route('/api/test', methods=['GET'])
def test_api():
    return jsonify({"message": "API is working!", "status": "success"})

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    try:
        print("Received request at /api/chat")
        data = request.json
        print(f"Request data: {data}")

        user_message = data['message']

        # Build message objects (system + history)
        message_objects = [system_message]
        for turn in chat_history["messages"]:
            message_objects.append(HumanMessage(content=turn["human"]))
            message_objects.append(AIMessage(content=turn["ai"]))

        # Add new user input
        message_objects.append(HumanMessage(content=user_message))

        # Get AI response
        result = chat_model.invoke(message_objects)

        # Save turn in dictionary
        chat_history["messages"].append({
            "human": user_message,
            "ai": result.content
        })

        return jsonify({'response': result.content})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'response': 'Sorry, I encountered an error processing your request.'})

@app.route('/api/clear', methods=['POST'])
def clear_chat():
    chat_history["messages"] = []
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)