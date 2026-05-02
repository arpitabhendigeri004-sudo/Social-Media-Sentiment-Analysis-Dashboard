import joblib

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict_sentiment(text):
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

# Test
if __name__ == "__main__":
    text = input("Enter text: ")
    print("Sentiment:", predict_sentiment(text))