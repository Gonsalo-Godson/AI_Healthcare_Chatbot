"""
📘 Re-train the AI Healthcare Chatbot model locally.
Run this file once to (re)generate your edge inference model:
    python train_model.py
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ---------------------------------------------
# Configuration
# ---------------------------------------------
BASE = Path(__file__).parent
DATA = BASE / "data_synthetic.csv"

SYMPTOMS = [
    "fever", "cough", "sore_throat", "runny_nose", "headache", "fatigue",
    "nausea", "vomiting", "diarrhea", "abdominal_pain", "chest_pain",
    "shortness_of_breath", "dizziness", "leg_swelling", "bleeding", "rash",
    "joint_pain", "loss_of_smell", "loss_of_taste", "sore_eyes"
]

# Common diseases we'll train on
DISEASES = [
    "Common Cold", "Influenza", "Gastroenteritis", "Migraine",
    "Deep Vein Thrombosis", "Allergic Reaction", "Conjunctivitis",
    "Hypertension Emergency", "Myocardial Infarction", "COVID-19"
]


def generate_synthetic_data(n_samples=500):
    """Generate a synthetic but realistic dataset for edge training."""
    np.random.seed(42)
    rows = []
    for _ in range(n_samples):
        condition = np.random.choice(DISEASES)
        symptoms = dict.fromkeys(SYMPTOMS, 0)

        # Assign realistic symptom patterns per disease
        if condition == "Common Cold":
            for s in ["fever", "cough", "runny_nose", "sore_throat", "fatigue"]:
                symptoms[s] = np.random.choice([0, 1], p=[0.2, 0.8])
        elif condition == "Influenza":
            for s in ["fever", "fatigue", "headache", "sore_throat", "cough"]:
                symptoms[s] = np.random.choice([0, 1], p=[0.3, 0.7])
        elif condition == "Gastroenteritis":
            for s in ["nausea", "vomiting", "diarrhea", "abdominal_pain"]:
                symptoms[s] = np.random.choice([0, 1], p=[0.3, 0.7])
        elif condition == "Migraine":
            for s in ["headache", "nausea", "dizziness"]:
                symptoms[s] = np.random.choice([0, 1], p=[0.4, 0.6])
        elif condition == "Deep Vein Thrombosis":
            for s in ["leg_swelling", "chest_pain", "shortness_of_breath"]:
                symptoms[s] = np.random.choice([0, 1], p=[0.3, 0.7])
        elif condition == "Allergic Reaction":
            for s in ["rash", "shortness_of_breath", "sore_eyes"]:
                symptoms[s] = np.random.choice([0, 1], p=[0.4, 0.6])
        elif condition == "Conjunctivitis":
            for s in ["sore_eyes", "headache", "fever"]:
                symptoms[s] = np.random.choice([0, 1], p=[0.3, 0.7])
        elif condition == "Hypertension Emergency":
            for s in ["headache", "dizziness", "chest_pain"]:
                symptoms[s] = np.random.choice([0, 1], p=[0.3, 0.7])
        elif condition == "Myocardial Infarction":
            for s in ["chest_pain", "shortness_of_breath", "fatigue"]:
                symptoms[s] = np.random.choice([0, 1], p=[0.3, 0.7])
        elif condition == "COVID-19":
            for s in ["fever", "cough", "fatigue", "loss_of_smell", "loss_of_taste"]:
                symptoms[s] = np.random.choice([0, 1], p=[0.3, 0.7])

        row = {"condition": condition}
        row.update(symptoms)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def main():
    print("🔄 Training new healthcare model...")

    # Create synthetic dataset if not present
    if not DATA.exists():
        print("📁 Generating synthetic dataset...")
        df = generate_synthetic_data(1000)
        df.to_csv(DATA, index=False)
        print(f"✅ Saved dataset at {DATA}")
    else:
        df = pd.read_csv(DATA)
        print(f"📄 Loaded dataset: {DATA}")

    # Prepare data
    X = df[SYMPTOMS].values
    le = LabelEncoder()
    y = le.fit_transform(df["condition"].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Training complete. Accuracy: {acc*100:.2f}%")
    print("\n📊 Classification Report:\n", classification_report(y_test, preds, target_names=le.classes_))

    # Save model and label encoder
    with open(BASE / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(BASE / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("\n💾 Model and label encoder saved successfully!")
    print("🚀 You can now run: python main.py or python app.py")


if __name__ == "__main__":
    main()
