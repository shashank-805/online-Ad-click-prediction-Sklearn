import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("data/online_ad_clicks.csv")

X = df.drop("clicked", axis=1)
y = df["clicked"]

categorical_features = ["gender", "device", "ad_position"]
numeric_features = [
    "age",
    "daily_time_on_site",
    "pages_visited",
    "annual_income",
    "internet_usage_hours",
    "previous_clicks"
]

preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("numeric", "passthrough", numeric_features)
])

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nClassification Report:\n")
print(classification_report(y_test, predictions, zero_division=0))

joblib.dump(pipeline, "online_ad_click_model.pkl")
print("Model saved to online_ad_click_model.pkl")

feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
importances = pipeline.named_steps["model"].feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False).head(15)

plt.figure(figsize=(10, 6))
plt.barh(
    importance_df["feature"][::-1],
    importance_df["importance"][::-1]
)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top Features for Online Ad Click Prediction")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Feature importance chart saved to feature_importance.png")
