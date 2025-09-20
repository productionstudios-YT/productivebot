# app.py
from fastapi import FastAPI, Request
import tensorflow as tf
import numpy as np

# Optional: for text preprocessing when you add tokenizer later
# from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

# Load your trained model
model = tf.keras.models.load_model("chat_model.h5")

# Optional: load tokenizer and MAX_LEN if you trained with one
# tokenizer = ...  
# MAX_LEN = ...

@app.post("/predict")
async def predict(req: Request):
    data = await req.json()
    text = data.get("queryResult", {}).get("queryText", "").strip()

    # Simple greetings logic before model preprocessing
    greetings = ["hi", "hello", "hey"]
    if text.lower() in greetings:
        response = "Hello! How can I help you today?"
    else:
        # Replace the following with your actual tokenizer logic
        # Example:
        # tokens = tokenizer.texts_to_sequences([text])
        # tokens = pad_sequences(tokens, maxlen=MAX_LEN)
        # prediction = float(model.predict(tokens)[0][0])

        # Temporary placeholder using random input
        tokens = np.random.rand(1, 10)
        prediction = float(model.predict(tokens)[0][0])
        response = "Positive" if prediction > 0.5 else "Negative"

    return {"fulfillmentText": f"🤖 ProductiveBot thinks: {response}"}

@app.get("/")
def home():
    return {"message": "ProductiveBot API is running!"}
