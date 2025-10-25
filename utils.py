"""
Utilities for AI Healthcare Chatbot
Contains unique recommendations for each disease and smart emergency detection.
"""

import pickle, os

SYMPTOMS = [
    "fever", "cough", "sore_throat", "runny_nose", "headache", "fatigue",
    "nausea", "vomiting", "diarrhea", "abdominal_pain", "chest_pain",
    "shortness_of_breath", "dizziness", "leg_swelling", "bleeding", "rash",
    "joint_pain", "loss_of_smell", "loss_of_taste", "sore_eyes"
]


def vectorize_symptoms(tokens):
    tokset = set(tokens)
    return [1 if s in tokset else 0 for s in SYMPTOMS]


def load_model(model_path, le_path):
    if not os.path.exists(model_path) or not os.path.exists(le_path):
        raise FileNotFoundError("Model or label encoder not found. Run train_model.py first.")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(le_path, "rb") as f:
        le = pickle.load(f)
    return model, le


def pretty_print_predictions(preds):
    print("\n🧩 Predicted Conditions (Top 3):")
    for cond, p in preds:
        print(f" - {cond:<25} {p*100:.1f}% confidence")


def emergency_check(tokens):
    """Detect high-risk symptoms not covered by model"""
    emergency_terms = {
        "cardiac_arrest", "heart_attack", "stroke", "severe_bleeding",
        "unconscious", "loss_of_consciousness", "difficulty_breathing",
        "severe_chest_pain"
    }
    for t in tokens:
        if t in emergency_terms:
            return True, t
    if "chest_pain" in tokens and ("shortness_of_breath" in tokens or "dizziness" in tokens):
        return True, "chest_pain + shortness_of_breath"
    return False, None


def get_emergency_advice(term):
    """Emergency recommendations"""
    term = term.lower()
    if "cardiac" in term or "heart" in term:
        return (
            "🚑 Possible cardiac arrest or heart attack.\n"
            "• Call emergency services (108 / 112) immediately.\n"
            "• Begin CPR if the person is unresponsive and not breathing.\n"
            "• Use an AED if available.\n"
            "• Do NOT give food or medication unless advised by a doctor."
        )
    if "stroke" in term:
        return (
            "🚑 Possible stroke detected.\n"
            "• Call emergency services immediately.\n"
            "• Note the time symptoms began.\n"
            "• Do NOT give food or water; keep patient calm and upright."
        )
    if "bleeding" in term:
        return (
            "🚑 Severe bleeding detected.\n"
            "• Apply firm pressure with a clean cloth.\n"
            "• Do not remove soaked cloth—add another on top.\n"
            "• Seek emergency medical attention immediately."
        )
    if "unconscious" in term:
        return (
            "🚨 Unconscious person detected.\n"
            "• Check breathing and pulse.\n"
            "• If absent, begin CPR and call emergency services.\n"
            "• Keep airway open and stay with the person."
        )
    if "difficulty_breathing" in term:
        return (
            "🚨 Severe breathing difficulty detected.\n"
            "• Sit upright and loosen tight clothing.\n"
            "• Use inhaler if prescribed.\n"
            "• Call emergency services right away."
        )
    return "🚨 Critical condition detected. Seek emergency medical help immediately."


def get_recommendations(condition):
    """Unique condition-specific advice"""
    condition = condition.lower()
    recs = {
        "common cold": (
            "• Rest and drink plenty of warm fluids.\n"
            "• Inhale steam to relieve nasal congestion.\n"
            "• Use saline nasal drops if nose is blocked.\n"
            "• Avoid cold drinks and dust exposure.\n"
            "• Usually resolves within a week."
        ),
        "influenza": (
            "• Get complete bed rest and stay warm.\n"
            "• Drink fluids frequently to avoid dehydration.\n"
            "• Take prescribed antiviral medication if advised.\n"
            "• Avoid public places until fever subsides.\n"
            "• Consult a doctor if symptoms persist beyond 5 days."
        ),
        "gastroenteritis": (
            "• Drink Oral Rehydration Solution (ORS) to replace lost fluids.\n"
            "• Avoid milk, spicy, and oily foods.\n"
            "• Eat bland items like rice, toast, and bananas.\n"
            "• Wash hands thoroughly to prevent reinfection.\n"
            "• See a doctor if vomiting persists or blood appears in stool."
        ),
        "migraine": (
            "• Rest in a dark and quiet room.\n"
            "• Apply a cold compress to your forehead.\n"
            "• Stay hydrated and maintain consistent sleep.\n"
            "• Avoid strong smells, caffeine, and loud noise.\n"
            "• Consult a neurologist if migraines are frequent."
        ),
        "hypertension emergency": (
            "🚨 Medical emergency!\n"
            "• Sit calmly and avoid exertion.\n"
            "• Do not take extra doses of medication unless prescribed.\n"
            "• Get immediate hospital evaluation.\n"
            "• Monitor blood pressure continuously until help arrives."
        ),
        "myocardial infarction": (
            "🚨 Suspected heart attack!\n"
            "• Call emergency services immediately.\n"
            "• Chew aspirin if prescribed by your doctor.\n"
            "• Sit down and stay calm.\n"
            "• Do NOT drive yourself to the hospital."
        ),
        "allergic reaction": (
            "• Identify and avoid the triggering allergen.\n"
            "• Take an antihistamine (e.g., cetirizine) if mild.\n"
            "• If swelling or breathing issues occur, use epinephrine if prescribed.\n"
            "• Visit an emergency room if symptoms worsen."
        ),
        "deep vein thrombosis": (
            "• Avoid sitting for long periods.\n"
            "• Keep your leg elevated when resting.\n"
            "• Do not massage the swollen area.\n"
            "• Consult a doctor for ultrasound and medication.\n"
            "• Regularly move or stretch if you sit for long hours."
        ),
        "covid-19": (
            "• Isolate yourself immediately to prevent transmission.\n"
            "• Monitor oxygen saturation and temperature.\n"
            "• Drink warm fluids and get adequate rest.\n"
            "• Seek medical attention if SpO2 < 94%.\n"
            "• Follow public health guidelines for quarantine."
        ),
        "conjunctivitis": (
            "• Wash hands frequently and avoid touching your eyes.\n"
            "• Use antibiotic or lubricating eye drops as prescribed.\n"
            "• Do not share towels, makeup, or contact lenses.\n"
            "• Avoid bright light and rest your eyes.\n"
            "• Usually resolves within 3–5 days."
        ),
    }

    return recs.get(
        condition,
        "• Rest, hydrate, and monitor symptoms closely.\n• Visit a healthcare professional if you feel unwell."
    )
