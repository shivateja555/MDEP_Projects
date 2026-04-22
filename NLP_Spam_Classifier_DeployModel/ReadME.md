# 🔍 NLP Spam Classifier with Explainability

An end-to-end Natural Language Processing (NLP) application that classifies text messages as **spam or ham (not spam)**.  
The system exposes a **REST API**, provides a **web-based user interface**, and generates **human-readable explanations** for every prediction.

The entire application is **containerized using Docker**, making it easy to run anywhere.

---

## 📁 Project Structure


nlp-spam-app/
├── src/
│ ├── app.py # Flask application (API + Web UI)
│ ├── model.py # Training and prediction logic
│ ├── explain.py # Explainability (feature importance)
│ ├── templates/
│ │ └── Index.html # Frontend UI
│ └── init.py # Makes src a Python package
├── data/
│ └── spamEmails.csv # Dataset (spam=1, ham=0)
├── models/
│ └── spam_model.pkl # Saved trained model (auto-created)
├── Dockerfile # Multi-stage Docker build
├── requirements.txt # Python dependencies
└── README.md


---

## 🚀 How the Project Works

1. The dataset is loaded from `data/spamEmails.csv`
2. A machine learning pipeline is built using:
   - **TF-IDF Vectorizer** → converts text into numerical features
   - **Logistic Regression** → performs classification
3. The trained model is saved in `models/spam_model.pkl`
4. The Flask app:
   - Loads the model
   - Accepts user input
   - Returns prediction + explanation
5. The frontend (HTML page):
   - Sends request to API
   - Displays prediction, confidence, and key words

---

## 🚀 Quick Start with Docker

1. Build the Docker image
docker build -t nlp-spam-app .
2. Run the container
docker run -p 5000:5000 nlp-spam-app
3. Open in browser
http://localhost:5000

## 💻 Run Locally (Without Docker)
1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
2. Install dependencies
pip install -r requirements.txt
3. Start the application
python src/app.py

⚠️ If the model file is not present, it will be automatically trained on first run.

## 🌐 Web Application

Open:
http://localhost:5000

Features:
Enter or paste any message
Detect whether it is Spam or Ham
View confidence score
See top words influencing prediction
Clean and interactive UI

## 📡 API Endpoints
### 🔹 1. Health Check

GET /health

curl http://localhost:5000/health

Response:

{
  "status": "ok",
  "model_loaded": true
}

### 🔹 2. Predict Single Message

POST /predict

Request:
{
  "text": "Congratulations! You have won a free prize!"
}

Example:
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Congratulations! You have won a free prize!"}'

Response:
{
  "text": "Congratulations! You have won a free prize!",
  "label": "spam",
  "probability": {
    "spam": 0.97,
    "ham": 0.03
  },
  "explanation": {
    "method": "feature_importance",
    "label": "spam",
    "top_words": [
      {"word": "won", "score": 0.42},
      {"word": "free", "score": 0.38},
      {"word": "prize", "score": 0.35}
    ],
    "explanation": "This message was classified as spam mainly because of the following words: won, free, prize."
  }
}

### 🔹 3. Batch Prediction

POST /predict/batch

curl -X POST http://localhost:5000/predict/batch \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Win money now!", "Are we meeting today?"]}'

## 🧠 Model Details
Component	Description
Vectorizer	TF-IDF (10,000 features, unigrams + bigrams)
Algorithm	Logistic Regression
Library	scikit-learn
Accuracy	~97% (depends on data split)

## 🔍 Explainability
The model uses Feature Importance to explain predictions.

How it works:

For each word:

Importance = TF-IDF score × Model coefficient
Output:
Top contributing words
Their influence scores
Human-readable explanation
Example:
"This message was classified as spam mainly because of:
won, free, prize"

✔ Fast
✔ Deterministic
✔ Always available

## 📦 Dataset
Source: SMS Spam Dataset (Kaggle)
Format:
text → message content
spam → 1 (spam), 0 (ham)

## ⚙️ Docker Details
Uses multi-stage build for smaller image
Python 3.11 slim base image
Installs dependencies in builder stage
Copies only required files to runtime image

## 📋 Environment
Python 3.11
Flask 3.0
scikit-learn 1.5
pandas 2.2
numpy 1.26
Docker

## ⚠️ Important Notes
src/ is treated as a Python package (__init__.py required ✔)
Templates must be inside src/templates/
PYTHONPATH=/app is set in Docker to resolve imports
Model is automatically trained if not found

## 🚀 Future Improvements
Add LIME explainability
Add authentication for API
Deploy using Gunicorn + Nginx
Add cloud deployment (AWS / GCP / Azure)
Add logging & monitoring
