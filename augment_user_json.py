import json
import random

# Load your existing user.json
with open("user.json", "r", encoding="utf-8") as f:
    user_intents = json.load(f)

# Slang/paraphrase options
slang_variants = {
    "you": ["u", "ya"],
    "your": ["ur"],
    "favorite": ["fav"],
    "do you": ["dya", "do ya"],
    "what is": ["whats", "what's"],
    "are you": ["r u"],
    "hello": ["yo", "sup"],
    "hi": ["hey", "yo"]
}

def add_variants(phrase):
    variants = [phrase]  # always keep original
    words = phrase.split()
    for i, w in enumerate(words):
        lw = w.lower()
        if lw in slang_variants:
            # randomly pick 1 alt (instead of all)
            alt = random.choice(slang_variants[lw])
            new = words[:i] + [alt] + words[i+1:]
            variants.append(" ".join(new))
    return list(set(variants))  # unique only

# Expand all intents
augmented = {}
for intent, examples in user_intents.items():
    expanded = []
    for ex in examples:
        expanded.extend(add_variants(ex))
    augmented[intent] = expanded

# Save as new file
with open("user_augmented.json", "w", encoding="utf-8") as f:
    json.dump(augmented, f, indent=2, ensure_ascii=False)

print("✅ Expanded user.json into user_augmented.json with *balanced* slang")
