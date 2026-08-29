# updated app.py

from flask import Flask, render_template,request
import mlflow
import pickle
import os
import pandas as pd

import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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


# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "sujat-khan"
repo_name = "MLFlow-DVC-Dagshub-Project"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

app = Flask(__name__)

# load model from model registry
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    all_versions = client.search_model_versions(f"name='{model_name}'")
    for stage in ["Production", "Staging", "None"]:
        stage_versions = [v for v in all_versions if v.current_stage == stage]
        if stage_versions:
            return max(stage_versions, key=lambda v: int(v.version)).version
    return None

model_name = "my_model"
model_version = get_latest_model_version(model_name)

model = None
if model_version:
    try:
        model_uri = f'models:/{model_name}/{model_version}'
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Successfully loaded model from DagsHub Model Registry: {model_uri}")
    except Exception as e:
        print(f"Could not load model version {model_version} from DagsHub ({e}).")
        print("Falling back to local models/model.pkl...")

if model is None:
    local_model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')
    if not os.path.exists(local_model_path):
        local_model_path = 'models/model.pkl'
    with open(local_model_path, 'rb') as f:
        model = pickle.load(f)
    print("Successfully loaded local models/model.pkl")

# Load vectorizer safely from project models/
vectorizer_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'vectorizer.pkl')
if not os.path.exists(vectorizer_path):
    vectorizer_path = 'models/vectorizer.pkl'
with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html',result=None)

@app.route('/predict', methods=['POST'])
def predict():

    text = request.form['text']

    # clean
    text = normalize_text(text)

    # bow
    features = vectorizer.transform([text])

    # prediction
    if hasattr(model, 'predict_proba'):
        # Raw sklearn model
        result = model.predict(features)
    else:
        # MLflow pyfunc model needs DataFrame input
        features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])
        result = model.predict(features_df)

    # show
    return render_template('index.html', result=int(result[0]))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")