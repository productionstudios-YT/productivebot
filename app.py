from fastapi import FastAPI, Request
import tensorflow as tf
import numpy as np

app = FastAPI()

# Load your trained model
model = tf.keras.models.load_model("chat_model.h5")

@app.post("/predict")
async def predict(req: Request):
    data = await req.json()
    text = data.get("queryResult", {}).get("queryText", "")
    
    # Ensure lowercase for comparison
    text_lower = text.strip().lower()

    # Simple greetings logic
    greetings = ["hi", "hello", "hey"]
    if any(greet == text_lower for greet in greetings):
        response = "Hello! How can I help you today?"
    else:
        # Temporary placeholder for model prediction
        tokens = np.random.rand(1, 10)
        prediction = float(model.predict(tokens)[0][0])
        response = "Positive" if prediction > 0.5 else "Negative"

    return {"fulfillmentText": f"🤖 ProductiveBot thinks: {response}"}

@app.get("/")
def home():
    return {"message": "ProductiveBot API is running!"}
