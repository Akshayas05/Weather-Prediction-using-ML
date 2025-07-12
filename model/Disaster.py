import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data\\Disaster.csv")

df = df.dropna(subset=['Title', 'Disaster_Info'])

def extract_disaster_type(title):
    title = title.lower()
    if "earthquake" in title:
        return "Earthquake"
    elif "flood" in title:
        return "Flood"
    elif "cyclone" in title or "typhoon" in title:
        return "Cyclone"
    elif "drought" in title:
        return "Drought"
    elif "landslide" in title:
        return "Landslide"
    elif "storm" in title or "hurricane" in title:
        return "Storm"
    else:
        return "Other"

df['Disaster_Type'] = df['Title'].apply(extract_disaster_type)

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['Cleaned_Info'] = df['Disaster_Info'].apply(clean_text)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Cleaned_Info'])
y = df['Disaster_Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Disaster Type Classification")
plt.tight_layout()
plt.savefig("disaster_confusion_matrix.png")
plt.show()

import joblib
import os

os.makedirs("models", exist_ok=True)
joblib.dump(vectorizer, "models//disaster_vectorizer.pkl")
joblib.dump(model, "models//disaster_model.pkl")
print("Disaster model and vectorizer saved.")

