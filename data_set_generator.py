import os
import json
from datasets import load_dataset

# -----------------
# Load profile facts
# -----------------
with open("profile.json", "r", encoding="utf-8") as f:
    profile = json.load(f)

# Convert profile dict into a marked-up string
profile_text = "<|profile|>\n"
for key, value in profile.items():
    if isinstance(value, list):
        value_str = ", ".join(map(str, value))
    else:
        value_str = str(value)
    profile_text += f"- {key}: {value_str}\n"
profile_text += "<|endprofile|>\n\n"

# -----------------
# Function to append DailyDialog
# -----------------
def append_dailydialog_to_json(file_path: str = "chat_data.json"):
    """
    Appends DailyDialog conversations to chat_data.json.
    Each conversation includes profile facts wrapped in markers.
    """
    # Load existing dataset
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                data = json.loads(content) if content else []
        except json.JSONDecodeError:
            print("⚠️ JSON decode error, starting fresh.")
            data = []
    else:
        data = []

    # Load DailyDialog
    print("📥 Downloading DailyDialog dataset...")
    dataset = load_dataset("daily_dialog")

    new_dialogues = []
    for split in ["train", "validation", "test"]:
        for conv in dataset[split]:
            dialogue = [profile_text]  # inject profile at the start
            for idx, utterance in enumerate(conv["dialog"]):
                if idx % 2 == 0:
                    dialogue.append(f"User: {utterance}")
                else:
                    dialogue.append(f"Bot: {utterance}")
            new_dialogues.append({"dialogue": dialogue})

    # Append and save
    data.extend(new_dialogues)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Added {len(new_dialogues)} DailyDialog conversations with profile facts.")
    print(f"📊 Total conversations in {file_path}: {len(data)}")

# -----------------
# Run if executed
# -----------------
if __name__ == "__main__":
    append_dailydialog_to_json("chat_data.json")
