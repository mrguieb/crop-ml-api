import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1) Load dataset
df = pd.read_csv("data/crop_recommendation.csv")  # Kaggle/GitHub CSV
# Expected columns: N,P,K,temperature,humidity,ph,rainfall,label

# 2) Encode labels
le = LabelEncoder()
y = le.fit_transform(df["label"])

X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]

# 3) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Pipeline: scale (for temperature etc.) + RandomForest
pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),  # RF doesn’t need scaling, but harmless
    ("rf", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        n_jobs=-1,
        random_state=42
    ))
])

pipe.fit(X_train, y_train)
print(classification_report(y_test, pipe.predict(X_test), target_names=le.classes_))

# 5) Persist
joblib.dump(pipe, "model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("Saved model.pkl and label_encoder.pkl")
