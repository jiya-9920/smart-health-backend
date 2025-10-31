import pickle
import random

# Simple rule-based model (pretending ML for demo)
class HealthModel:
    def predict(self, symptoms):
        symptoms = [s.lower() for s in symptoms]
        if "fever" in symptoms and "cough" in symptoms:
            return "You might have the Common Cold 🤧"
        elif "headache" in symptoms and "nausea" in symptoms:
            return "You might be experiencing Migraine 😖"
        elif "stomach pain" in symptoms or "vomiting" in symptoms:
            return "You might have a Stomach Infection 🤢"
        elif "fatigue" in symptoms:
            return "You might be suffering from Fatigue 😴"
        else:
            return random.choice([
                "Looks like you're healthy! 😊",
                "Minor symptoms — take rest and stay hydrated 💧",
                "Consult a doctor if symptoms persist 🏥"
            ])

if __name__ == "__main__":
    model = HealthModel()
    with open("models/health_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("✅ Real Health Model saved successfully!")
