from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import dagshub
import numpy as np

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)

    return text

# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/Dhruvkulshrestha018/MLOps-project.mlflow')
# dagshub.init(repo_owner='Dhruvkulshrestha018', repo_name='MLOps-project', mlflow=True)
# -------------------------------------------------------------------------------------

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Dhruvkulshrestha018"
repo_name = "MLOps-project"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------


# Initialize Flask app
app = Flask(__name__)

# Create a custom registry
registry = CollectorRegistry()

# Define your custom metrics using this registry
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)

# ------------------------------------------------------------------------------------------
# Model and vectorizer setup with lazy loading
model_name = "my_model"

def get_latest_model_version(model_name):
    """Get the latest version of the model from MLflow."""
    try:
        client = mlflow.MlflowClient()
        # First try to get staging versions
        latest_version = client.get_latest_versions(model_name, stages=["staging"])
        if not latest_version:
            # If no staging versions, get versions with no stage
            latest_version = client.get_latest_versions(model_name, stages=["None"])
        return latest_version[0].version if latest_version else None
    except Exception as e:
        print(f"Error getting latest model version: {e}")
        return None

def get_model():
    """Lazy load the MLflow model."""
    if not hasattr(app, 'model'):
        model_version = get_latest_model_version(model_name)
        if model_version:
            model_uri = f'models:/{model_name}/{model_version}'
            print(f"Fetching model from: {model_uri}")
            try:
                app.model = mlflow.pyfunc.load_model(model_uri)
            except Exception as e:
                print(f"Error loading model: {e}")
                app.model = None
        else:
            print(f"No model version found for '{model_name}'")
            app.model = None
    return app.model

def get_vectorizer():
    """Lazy load the vectorizer."""
    if not hasattr(app, 'vectorizer'):
        try:
            # Check if vectorizer file exists
            vectorizer_path = 'models/vectorizer.pkl'
            if os.path.exists(vectorizer_path):
                app.vectorizer = pickle.load(open(vectorizer_path, 'rb'))
            else:
                print(f"Vectorizer file not found at {vectorizer_path}")
                app.vectorizer = None
        except Exception as e:
            print(f"Error loading vectorizer: {e}")
            app.vectorizer = None
    return app.vectorizer

# Routes
@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None, error=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    try:
        text = request.form["text"]
        if not text:
            return render_template("index.html", result=None, error="Please enter some text")
        
        # Clean text
        text = normalize_text(text)
        
        # Get model and vectorizer lazily
        vectorizer = get_vectorizer()
        model = get_model()
        
        # Check if model and vectorizer are loaded
        if vectorizer is None:
            return render_template("index.html", result=None, error="Vectorizer not loaded. Please check system configuration.")
        
        if model is None:
            return render_template("index.html", result=None, error="Model not loaded. Please check system configuration.")
        
        # Convert to features
        features = vectorizer.transform([text])
        features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

        # Predict
        result = model.predict(features_df)
        prediction = result[0]

        # Increment prediction count metric
        PREDICTION_COUNT.labels(prediction=str(prediction)).inc()

        # Measure latency
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

        return render_template("index.html", result=prediction, error=None)
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template("index.html", result=None, error=f"Prediction error: {str(e)}")

@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose only custom Prometheus metrics."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    model = get_model()
    vectorizer = get_vectorizer()
    
    status = {
        "status": "healthy" if model is not None and vectorizer is not None else "degraded",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    }
    return status, 200 if status["status"] == "healthy" else 503

if __name__ == "__main__":
    # For local use, you might want to pre-load the model
    if os.getenv("PRELOAD_MODEL", "false").lower() == "true":
        print("Pre-loading model and vectorizer...")
        get_model()
        get_vectorizer()
    
    # app.run(debug=True) # for local use
    app.run(debug=True, host="0.0.0.0", port=5000)  # Accessible from outside Docker