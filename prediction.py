import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/matches.csv")

# Remove rows where winner is missing
df = df.dropna(subset=['winner'])

# Select features
features = ['team1', 'team2', 'toss_winner', 'toss_decision', 'venue']
target = 'winner'

df_model = df[features + [target]].dropna()

# Encode text to numbers
le = LabelEncoder()
for col in features + [target]:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

X = df_model[features]
y = df_model[target]

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Check accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("ML Model trained successfully!")
import pickle
import os

# Save the model
os.makedirs("models", exist_ok=True)

with open("models/win_predictor.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")

# Predict a match - Example: CSK vs MI
print("\n=== Match Prediction ===")
sample = X_test.iloc[0]
result = model.predict([sample])
print("Prediction done! Model is working!")