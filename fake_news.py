import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import os

files = [
    "C:/Users/hp5cd/OneDrive/Desktop/Fake_News/WELFake_Dataset.csv.xlsx",
    "C:/Users/hp5cd/OneDrive/Desktop/Fake_News/FakeNewsNet.csv.xlsx",
    "C:/Users/hp5cd/OneDrive/Desktop/Fake_News/fake.xlsx"

]

def load_and_standardize(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    df.columns = [c.lower() for c in df.columns]

    title = next((c for c in df.columns if "title" in c or "headline" in c), None)
    body = next((c for c in df.columns if "text" in c or "content" in c or "article" in c), None)
    label = next((c for c in df.columns if "label" in c or "class" in c or "truth" in c), None)

    data = pd.DataFrame()
    data["title"] = df[title].astype(str) if title else ""
    data["body"] = df[body].astype(str) if body else ""
    data["text"] = (data["title"] + " " + data["body"]).fillna("")

    def map_label(x):
        x = str(x).lower()
        if "fake" in x or "false" in x or x == "1":
            return 1
        elif "real" in x or "true" in x or x == "0":
            return 0
        else:
            return np.nan

    data["label"] = df[label].map(map_label) if label else np.nan
    return data

datasets = [load_and_standardize(f) for f in files]
data = pd.concat(datasets, ignore_index=True).dropna(subset=["label"]).reset_index(drop=True)
print(f"✅ Total Combined Records: {data.shape[0]}")

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = text.lower().strip()
    return text

data["clean_text"] = data["text"].apply(clean_text)

def extract_meta_features(df):
    features = pd.DataFrame()
    features["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))
    features["char_count"] = df["clean_text"].apply(len)
    features["avg_word_len"] = features["char_count"] / (features["word_count"] + 1)
    features["punct_count"] = df["text"].apply(lambda x: len(re.findall(r'[^\w\s]', x)))
    features["uppercase_ratio"] = df["text"].apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x)+1))
    return features

meta_features = extract_meta_features(data)

tfidf_small = TfidfVectorizer(max_features=500)
title_vec = tfidf_small.fit_transform(data["title"])
body_vec = tfidf_small.transform(data["body"])
cosine_sim = [
    cosine_similarity(title_vec[i], body_vec[i])[0][0] if title_vec[i].nnz and body_vec[i].nnz else 0
    for i in range(title_vec.shape[0])
]
data["cosine_similarity"] = cosine_sim

X_text = data["clean_text"]
y = data["label"].astype(int)

X_train_text, X_test_text, y_train, y_test, meta_train, meta_test = train_test_split(
    X_text, y, meta_features, test_size=0.2, random_state=42
)

tfidf = TfidfVectorizer(max_features=8000, stop_words="english")
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

X_train = hstack([X_train_tfidf, meta_train.values])
X_test = hstack([X_test_tfidf, meta_test.values])

lr_model = LogisticRegression(max_iter=500)
lr_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=150, random_state=42)
rf_model.fit(X_train, y_train)

for name, model in [("Logistic Regression", lr_model), ("Random Forest", rf_model)]:
    preds = model.predict(X_test)
    print(f"\n📊 {name} Results:")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, digits=4))

# Save models
joblib.dump({"model": lr_model, "tfidf": tfidf}, "fake_news_lr.pkl")
joblib.dump({"model": rf_model, "tfidf": tfidf}, "fake_news_rf.pkl")
print("\n✅ Models saved successfully!")
