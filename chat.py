from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load a small pretrained chatbot model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

chat_history_ids = None

print("🤖 Chatbot is ready! Type 'quit' to exit.")

while True:
    user_input = input("You: ")
    if user_input.lower() in {"quit", "exit"}:
        print("Bot: Goodbye! 👋")
        break

    # Encode the user input and append to chat history
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    if chat_history_ids is not None:
        input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        input_ids = new_input_ids

    # Generate a response
    chat_history_ids = model.generate(
        input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,      # makes replies more natural
        top_p=0.1,
        temperature=0.1,
        repetition_penalty=1.0
    )

    # Decode and print the last generated tokens (the reply)
    reply = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"Bot: {reply}")
