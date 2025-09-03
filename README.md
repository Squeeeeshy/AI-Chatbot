# AI Chatbot (LoRA Fine-Tuned GPT-2)

This project is a **personalized AI chatbot** built on top of **GPT-2**, fine-tuned with **LoRA (Low-Rank Adaptation)**.  
The main goals of this project are:

- ✅ **Lightweight Training** – LoRA only trains a small number of additional parameters, so you don’t need massive GPUs.  
- ✅ **Custom Conversations** – Train the model with your own dialogues stored in `chat_data.json`.  
- ✅ **DailyDialog Dataset Integration** – Easily append a large, real-world conversation dataset to boost training.  
- ✅ **Separation of Profile Facts** – Profile data (like your school, hobbies, or work experience) is kept separate from training so it doesn’t “leak” into generic dialogues.  
- ✅ **Interactive Chatting** – Once trained, you can chat with the model in real time using `inference.py`.  

Under the hood:  
- The chatbot alternates between **User** and **Bot** roles in the dataset.  
- Each conversation is saved with an `<|endoftext|>` token so the model learns when a dialogue ends.  
- LoRA adapters are applied only to GPT-2’s attention layers (`c_attn`), making fine-tuning efficient while preserving GPT-2’s original knowledge.  

This makes the chatbot both **customizable** and **scalable**: you can add new datasets, fine-tune again, and keep improving the model without starting from scratch.

# Setup

## 🔧 Setup

```bash
1. Clone the repository
   git clone https://github.com/Squeeeeshy/AI-Chatbot.git
   cd AI-Chatbot

2. Create a virtual environment
   python -m venv .venv

3. Activate the environment
   # On Linux/Mac:
   source .venv/bin/activate
   # On Windows (PowerShell):
   .venv\Scripts\Activate

4. Install dependencies
   pip install -r requirements.txt

```
# Pipeline

## 🚀 Pipeline Procedure


```bash

1. Generate or append training data
   # Use your dataset generator (intent-based + profile-aware)
   python generate_dataset.py

   # (Optional) Add DailyDialog dataset to enrich conversations
   python append_dailydialog.py

   # The combined dataset will be saved in:
   chat_data.json

2. Train the chatbot
   # Fine-tune GPT-2 with LoRA on your dataset
   python train_chatbot.py

   # The trained model and tokenizer are saved into:
   ./gpt2_chatbot

3. Run the chatbot
   # Start an interactive session with the trained model
   python chatbot.py

   # Example usage:
   User: hi
   Bot: Hey! How's it going?

   User: what do you study?
   Bot: I'm in my 2B term studying Electrical Engineering at Waterloo.

4. (Optional) Update training data
   # If you want to add more dialogues later, re-run dataset generation
   python generate_dataset.py
   python append_dailydialog.py

   # Then retrain with:
   python train_chatbot.py
