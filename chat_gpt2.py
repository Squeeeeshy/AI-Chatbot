import torch
import random
import json
import re
from difflib import get_close_matches
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# -----------------------------
# Device setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

# -----------------------------
# Load fine-tuned GPT-2
# -----------------------------
model_path = "./gpt2_chatbot"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Ensure tokenizer has pad token
tokenizer.pad_token = tokenizer.eos_token
model.to(device)
model.eval()

# -----------------------------
# Load user intents for classification
# -----------------------------
with open("user_augmented.json", "r", encoding="utf-8") as f:
    user_intents = json.load(f)

# Precompute flat list of (utterance, intent)
intent_examples = []
for intent, examples in user_intents.items():
    for ex in examples:
        cleaned = re.sub(r"[^\w\s]", "", ex.lower()).strip()
        intent_examples.append((cleaned, intent))

def normalize(text):
    return re.sub(r"[^\w\s]", "", text.lower()).strip()

def classify_intent(user_input):
    cleaned = normalize(user_input)

    # Exact match
    for ex, intent in intent_examples:
        if ex == cleaned:
            return intent

    # Fuzzy match
    candidates = [ex for ex, _ in intent_examples]
    matches = get_close_matches(cleaned, candidates, n=1, cutoff=0.6)
    if matches:
        for ex, intent in intent_examples:
            if ex == matches[0]:
                return intent

    return None

print("🤖 Chatbot ready! Type 'quit' to exit.\n")

# -----------------------------
# Conversation loop
# -----------------------------
history = []  # Keep last few exchanges

while True:
    user_input = input("User: ").strip()
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Bot: Goodbye! 👋")
        break

    # Detect intent (Option B → append tag for GPT-2)
    intent = classify_intent(user_input)
    if intent:
        tagged_input = f"{user_input} [intent: {intent}]"
    else:
        tagged_input = user_input

    # Add tagged input to history
    history.append(f"User: {tagged_input}")

    # Keep only the last few exchanges (avoid history overflow)
    history = history[-6:]  # 3 user + 3 bot messages

    # Build prompt
    prompt = "\n".join(history) + "\nBot:"
    # print(f"\n[DEBUG PROMPT]\n{prompt}\n")  # <-- uncomment if you want to debug

    # Encode with proper attention mask
    input_enc = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512  # full GPT-2 context
    ).to(device)

    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            input_enc["input_ids"],
            attention_mask=input_enc["attention_mask"],
            max_new_tokens=60,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )

    # Decode generated response (skip the prompt part)
    generated_tokens = output_ids[0][input_enc["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Cleaner response processing
    if "User:" in response:
        response = response.split("\nUser:")[0].strip()
    else:
        response = response.split("\n")[0].strip()

    response = response.rstrip('.,!?;:')

    # Fallback if nonsense or empty
    if (not response or
        len(response.split()) < 1 or
        any(word in response.lower() for word in ['irl', 'ㅠ', '♡', '♥', '😅'])):

        fallbacks = [
            "I'm not sure how to respond to that. Could you ask something else?",
            "That's an interesting question. Could you rephrase it?",
            "I'm still learning about that topic. What else would you like to know?",
            "Let me think about that... Maybe ask me something else?",
            "I don't have a good answer for that right now. Try another question!"
        ]
        response = random.choice(fallbacks)

    # Add bot response to history
    history.append(f"Bot: {response}")

    print("Bot:", response)
