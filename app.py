from fastapi import FastAPI, Request
import tensorflow as tf
import numpy as np

app = FastAPI()

# Load trained model
model = tf.keras.models.load_model("chat_model.h5")

@app.post("/predict")
async def predict(req: Request):
    data = await req.json()
    text = data.get("queryResult", {}).get("queryText", "")

    # Fake preprocessing (replace with tokenizer later)
    tokens = np.random.rand(1, 10)

    prediction = float(model.predict(tokens)[0][0])
    response = "Positive" if prediction > 0.5 else "Negative"

    return {
        "fulfillmentText": f"🤖 ProductiveBot thinks: {response}"
    }

@app.get("/")
def home():
    return {"message": "ProductiveBot API is running!"}
