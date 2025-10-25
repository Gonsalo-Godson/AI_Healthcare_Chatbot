from flask import Flask, render_template, request, jsonify, session
import os
from datetime import timedelta
from utils import (
    SYMPTOMS,
    vectorize_symptoms,
    load_model,
    emergency_check,
    get_emergency_advice,
    get_recommendations,
)

app = Flask(__name__)
app.secret_key = "smart_healthcare_chatbot_edge"
app.permanent_session_lifetime = timedelta(minutes=5)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
LE_PATH = os.path.join(os.path.dirname(__file__), "label_encoder.pkl")
model, le = load_model(MODEL_PATH, LE_PATH)


@app.route("/")
def index():
    session.clear()
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip().lower()
    if not user_input:
        return jsonify({"reply": "Please describe your symptoms."})

    # 🩷 Friendly exit or gratitude handling
    if any(word in user_input for word in [
        "thank", "thanks", "thankyou", "bye", "goodbye", "see you", "take care"
    ]):
        session.clear()
        return jsonify({
            "reply": "You're most welcome! 🌿 Take care of your health and have a wonderful day!"
        })

    # 🧠 Restart conversation on greetings
    if user_input in ("hi", "hello", "hey", "start", "good morning", "good evening"):
        session.clear()
        session["step"] = "symptom"
        return jsonify({
            "reply": """
🤖 Hello again! I’m your AI-based Smart Healthcare Assistant.<br>
I can help you understand possible conditions based on your symptoms.<br>
Type 'list' to view all symptoms or 'exit' to quit anytime.<br><br>
Please tell me what symptoms you're experiencing:
"""
        })

    # Exit command
    if user_input in ("exit", "quit"):
        session.clear()
        return jsonify({"reply": "Take care of your health. Goodbye! 🩺"})

    # List all symptoms
    if user_input == "list":
        return jsonify({"reply": "🩺 Here are the symptoms I recognize:<br>" + ", ".join(SYMPTOMS)})

    step = session.get("step", "symptom")

    # Step 1 — Symptom input
    if step == "symptom":
        if user_input in ("no", "none", "nothing", "not really", "i’m fine", "i am fine", "feeling good", "nothing serious"):
            session.clear()
            return jsonify({
                "reply": "You mentioned no symptoms — that's great! 😊 Stay healthy and hydrated.<br>"
                         "If you ever feel unwell later, just type 'hi' to start again."
            })

        # 🧠 Extract symptoms from full sentence
        words = user_input.replace(",", " ").replace(".", "").split()
        tokens = [w.strip().replace(" ", "_") for w in words if w.strip()]
        matched = [t for t in tokens if t in SYMPTOMS]

        if not matched:
            return jsonify({
                "reply": "I’m not sure I recognized any symptoms there.<br>"
                         "Try describing your symptoms more clearly — for example: fever, sore throat, or fatigue."
            })

        # 🚨 Emergency detection
        emergency, matched_term = emergency_check(matched)
        if emergency:
            advice = get_emergency_advice(matched_term)
            return jsonify({
                "reply": f"🚨 That sounds serious — detected '{matched_term}'.<br>{advice}<br>⚕️ Please contact emergency services immediately."
            })

        session["tokens"] = matched
        session["step"] = "duration"
        return jsonify({"reply": "How long have you had these symptoms? (e.g., 2 days, 1 week)"})

    # Step 2 — Duration
    elif step == "duration":
        session["duration"] = user_input
        session["step"] = "severity"
        return jsonify({"reply": "On a scale of 1 to 10, how severe would you say they are?"})

    # Step 3 — Severity + Fever Rule
    elif step == "severity":
        session["severity"] = user_input
        tokens = session["tokens"]

        # 🧠 FEVER RULE — apply BEFORE prediction
        if "fever" in tokens and "leg_swelling" not in tokens:
            preds = [("Common Cold", 0.6), ("Influenza", 0.3), ("COVID-19", 0.1)]
        else:
            vec = vectorize_symptoms(tokens)
            proba = model.predict_proba([vec])[0]
            topk = sorted(list(enumerate(proba)), key=lambda x: x[1], reverse=True)[:3]
            preds = [(le.inverse_transform([i])[0], p) for i, p in topk]

        session["preds"] = preds
        session["step"] = "ask_recommendations"

        reply = "Here's what I found:<br>"
        for cond, p in preds:
            reply += f"➡️ {cond} — {p*100:.1f}% likelihood<br>"
        reply += "<br>Would you like my advice for these? (yes/no)"
        return jsonify({"reply": reply})

    # Step 4 — Recommendations
    elif step == "ask_recommendations":
        yes_words = {"yes", "y", "yeah", "sure", "ok", "okay", "of course", "please", "yes please"}
        no_words = {"no", "n", "nope", "not now"}

        if any(word in user_input for word in yes_words):
            preds = session.get("preds", [])
            reply = "💡 Recommendations:<br>"
            for cond, _ in preds:
                reply += f"<b>{cond}</b>: {get_recommendations(cond)}<br><br>"
            session["step"] = "more_symptoms"
            reply += "Would you like to describe any other symptoms? (yes/no)"
            return jsonify({"reply": reply})

        elif any(word in user_input for word in no_words):
            session["step"] = "more_symptoms"
            return jsonify({
                "reply": "Alright, no worries! I hope you feel better soon 💙<br>"
                         "Would you like to describe any other symptoms? (yes/no)"
            })

        else:
            return jsonify({"reply": "Please respond with 'yes' or 'no' 🙂"})

    # Step 5 — Continue or End
    elif step == "more_symptoms":
        if any(word in user_input for word in ["yes", "y", "sure", "ok", "okay"]):
            session["step"] = "symptom"
            return jsonify({"reply": "Please tell me what symptoms you're experiencing:"})
        else:
            session.clear()
            return jsonify({"reply": "Okay! Take rest and monitor your health. Bye 👋"})

    return jsonify({"reply": "I'm not sure I understood that. Could you repeat?"})


if __name__ == "__main__":
    app.run(debug=True)
