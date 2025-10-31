import pickle
import os

# A very simple dummy model class
class DummyModel:
    def predict(self, symptoms):
        if "fever" in symptoms or "cough" in symptoms:
            return "Common Cold"
        elif "headache" in symptoms:
            return "Migraine"
        else:
            return "Healthy"

# Create model instance
model = DummyModel()

# Save it inside models folder
os.makedirs("models", exist_ok=True)
with open("models/dummy_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Dummy model saved successfully!")
