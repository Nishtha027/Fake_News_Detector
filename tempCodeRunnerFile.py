joblib.dump({"model": lr_model, "tfidf": tfidf}, "fake_news_lr.pkl")
# joblib.dump({"model": rf_model, "tfidf": tfidf}, "fake_news_rf.pkl")
# print("\n✅ Models saved successfully!")

# # ----------------------------------------------------------
# # 🔟 Example Prediction Function
# # ----------------------------------------------------------
# def predict_news(text):
#     cleaned = clean_text(text)
#     text_vec = tfidf.transform([cleaned])
#     meta = extract_meta_features(pd.DataFrame({"clean_text": [cleaned], "text": [text]}))
#     X_final = hstack([text_vec, meta.values])
#     pred = lr_model.predict(X_final)[0]
#     label = "FAKE NEWS" if pred == 1 else "REAL NEWS"
#     return label

# # Example usage:
# example = "Breaking: Scientists discover a cure for aging using AI."
# print("\n📰 Example Prediction ->", predict_news(example))