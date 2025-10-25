"""
AI-Based Smart Healthcare Chatbot (Conversational Terminal Version)
"""

import os
import time
from utils import (
    SYMPTOMS,
    vectorize_symptoms,
    load_model,
    pretty_print_predictions,
    emergency_check,
    get_emergency_advice,
    get_recommendations,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
LE_PATH = os.path.join(os.path.dirname(__file__), "label_encoder.pkl")


def slow_print(text, delay=0.03):
    """Simulate natural chatbot typing."""
    for ch in text:
        print(ch, end='', flush=True)
        time.sleep(delay)
    print()


def get_duration():
    slow_print("How long have you had these symptoms? (e.g., 2 days, 1 week)")
    return input("You: ").strip()


def get_severity():
    slow_print("On a scale of 1 to 10, how severe would you say they are?")
    return input("You: ").strip()


def chat_loop(model, label_encoder):
    slow_print("🤖 Hello! I’m your AI-based Smart Healthcare Assistant.")
    slow_print("I can help you understand possible conditions based on your symptoms.")
    slow_print("Type 'list' to view all symptoms or 'exit' to quit anytime.\n")

    while True:
        slow_print("Please tell me what symptoms you're experiencing:")
        user = input("You: ").strip()
        if not user:
            continue
        if user.lower() in ("exit", "quit"):
            slow_print("Take care of your health. Goodbye! 🩺")
            break
        if user.lower() == "list":
            print(", ".join(SYMPTOMS))
            continue

        tokens = [t.strip().lower().replace(" ", "_") for t in user.split(",")]

        # Emergency detection
        emergency, matched_term = emergency_check(tokens)
        if emergency:
            slow_print(f"🚨 That sounds serious — detected '{matched_term}'.")
            slow_print(get_emergency_advice(matched_term))
            slow_print("⚕️ Please contact emergency services immediately.")
            continue

        duration = get_duration()
        severity = get_severity()
        slow_print("\nAnalyzing your symptoms...")
        time.sleep(1.5)

        # Prediction
        vec = vectorize_symptoms(tokens)
        proba = model.predict_proba([vec])[0]
        topk = sorted(list(enumerate(proba)), key=lambda x: x[1], reverse=True)[:3]
        preds = [(label_encoder.inverse_transform([i])[0], p) for i, p in topk]

        slow_print("\nHere's what I found:")
        for cond, p in preds:
            slow_print(f"➡️ {cond} — {p*100:.1f}% likelihood")

        slow_print("\nWould you like my advice for these?")
        confirm = input("You (yes/no): ").strip().lower()
        if confirm in ("yes", "y"):
            for cond, _ in preds:
                slow_print(f"\n💡 For {cond}:")
                slow_print(get_recommendations(cond))
        else:
            slow_print("Alright, no worries! I hope you feel better soon 💙")

        slow_print("\nWould you like to describe any other symptoms? (yes/no)")
        again = input("You: ").strip().lower()
        if again not in ("yes", "y"):
            slow_print("Okay! Take rest and monitor your health. Bye 👋")
            break


if __name__ == "__main__":
    model, le = load_model(MODEL_PATH, LE_PATH)
    chat_loop(model, le)
