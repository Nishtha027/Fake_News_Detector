import pandas as pd
import urllib.request
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------
df = pd.read_csv("combined_dataset.csv")

# Your dataset has "title" so we use that as text
df["text"] = df["title"].astype(str)

X = df["text"]
y = df["label"]


# ------------------------------------------------
# 2. TRAIN / TEST SPLIT
# ------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ------------------------------------------------
# 3. TF-IDF VECTOR
# ------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# ------------------------------------------------
# 4. TRAIN MODEL
# ------------------------------------------------
model = LogisticRegression(max_iter=500)
model.fit(X_train_vec, y_train)

# Show accuracy
preds = model.predict(X_test_vec)
print("ACCURACY:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# Save model
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("\nModel saved!\n")


# ========================================================
# 5. URL → TEXT EXTRACTION WITHOUT requests OR bs4
# ========================================================

def extract_text_from_url(url):
    try:
        # Use built-in urllib to fetch page
        response = urllib.request.urlopen(url)
        html = response.read().decode("utf-8", errors="ignore")

        # Remove script/style content
        html = re.sub(r"<script.*?>.*?</script>", " ", html, flags=re.DOTALL)
        html = re.sub(r"<style.*?>.*?</style>", " ", html, flags=re.DOTALL)

        # Extract all <p> paragraphs
        paragraphs = re.findall(r"<p.*?>(.*?)</p>", html, flags=re.DOTALL)

        cleaned = []
        for p in paragraphs:
            # Remove HTML tags
            p = re.sub(r"<.*?>", " ", p)
            p = re.sub(r"\s+", " ", p)
            cleaned.append(p.strip())

        return " ".join(cleaned)

    except Exception as e:
        print("Extraction error:", e)
        return ""


# ========================================================
# 6. PREDICT FROM URL
# ========================================================

def predict_from_url(url):
    print("Extracting article (NO requests, NO bs4)...")

    text = extract_text_from_url(url)

    if len(text) < 50:
        return "❌ Not enough text extracted."

    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    clean = re.sub(r"[^A-Za-z0-9 ]", " ", text.lower())
    vec = vectorizer.transform([clean])

    pred = model.predict(vec)[0]

    # 🔥 FIXED MAPPING
    pred_label = "REAL" if pred == 1 else "FAKE"

    return f"Prediction: {pred_label}"


# --------------------------------------------------------
# TEST URL
# --------------------------------------------------------
test_url = "https://stylecaster.com/beauty/lea-michele-hairstylist-second-day-hair-hack-coconut-oil-texture-spray/"
print(predict_from_url(test_url))
