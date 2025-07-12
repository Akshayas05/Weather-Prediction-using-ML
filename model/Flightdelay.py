import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("data\\Airline.csv")

label_encoder = LabelEncoder()
df['Airline'] = label_encoder.fit_transform(df['Airline'])
df['weather_desc'] = label_encoder.fit_transform(df['weather__hourly__weatherDesc__value'])

features = [
    'Airline', 'Distance', 'Departure Delay',
    'Passenger Load Factor', 'Airline Rating', 'Airport Rating',
    'Market Share', 'OTP Index',
    'weather__hourly__windspeedKmph', 'weather_desc',
    'weather__hourly__precipMM', 'weather__hourly__humidity',
    'weather__hourly__visibility', 'weather__hourly__pressure',
    'weather__hourly__cloudcover'
]
target = 'Category'
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}\n")
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png") 
plt.show()

importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")  
plt.show()

import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models\\flight_model.pkl")
print("Flight delay model saved.")
