from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random

# Load AI model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

chat_history_ids = None

# Intent dictionary
intents = {
    "greeting": ["hi", "hello", "hey", "good morning", "good evening"],
    "goodbye": ["bye", "goodbye", "see you", "later"],
    "weather": ["weather", "temperature", "forecast"],
    "joke": ["joke", "funny", "make me laugh"],
}

# Predefined responses
responses = {
    "greeting": ["Hello! 👋", "Hey there!", "Hi, how’s it going?"],
    "goodbye": ["Goodbye! 👋", "See you later!", "Take care!"],
    "weather": ["I can’t check live weather yet 🌦️, but it looks fine in here!"],
    "joke": ["Why don’t programmers like nature? Too many bugs 🐛😂"],
}


def detect_intent(user_input):
    user_input = user_input.lower()
    for intent, keywords in intents.items():
        if any(word in user_input for word in keywords):
            return intent
    return None


print("🤖 Chatbot with Intents ready! Type 'quit' to exit.")

while True:
    user_input = input("You: ")
    if user_input.lower() in {"quit", "exit"}:
        print("Bot: Goodbye! 👋")
        break

    # Step 1: Try to detect intent
    intent = detect_intent(user_input)
    if intent:
        reply = random.choice(responses[intent])
        print(f"Bot: {reply}")
        continue

    # Step 2: Fallback to AI model
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    if chat_history_ids is not None:
        input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        input_ids = new_input_ids

    chat_history_ids = model.generate(
        input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.2,
    )

    reply = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"Bot: {reply}")
