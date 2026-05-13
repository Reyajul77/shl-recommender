"""Run this once at deploy time to build the search index."""
import json, pickle
from sklearn.feature_extraction.text import TfidfVectorizer

with open("catalog.json") as f:
    catalog = json.load(f)

docs = []
for p in catalog:
    text = f"{p['name']} {p['description']} {' '.join(p['job_levels'])} {p['test_type']}"
    docs.append(text)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
matrix = vectorizer.fit_transform(docs)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("tfidf_matrix.pkl", "wb") as f:
    pickle.dump(matrix, f)

print(f"Built index for {len(docs)} products.")
