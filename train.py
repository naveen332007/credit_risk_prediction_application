import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data\\credit_risk_dataset.csv"
MODEL_PATH = "models\\model.joblib"
REPORT_PATH = "reports\\comparison.csv"


FEATURES=["annual_income", 
          "credit_score", 
          "loan_amount", 
          "existing_debt", 
          "late_payments", 
          "previous_defaults"
]

TARGET = "credit_risk"

df = pd.read_csv(DATA_PATH)
print("Rows in dataset:", len(df))
print("Risk Classes:", df[TARGET].unique())

X=df[FEATURES]
y=df[TARGET]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models={
    LogisticRegression: LogisticRegression(),
    DecisionTreeClassifier: DecisionTreeClassifier(),
    KNeighborsClassifier: KNeighborsClassifier(n_neighbors=5)
}

results = []
trained_models={}

for name, classifier in models.items():

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier",classifier)
    ])

    pipeline.fit(X_train, y_train)

    y_pred=pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    results.append({"models": name, "accuracy": round(accuracy,3)})
  
    trained_models[name] = pipeline


results_df=pd.DataFrame(results).sort_values("accuracy", ascending=False)

print(results_df.to_string(index=False))

best_name = results_df.iloc[0]["models"]
best_model = trained_models[best_name]
print(f"\nBest model: {best_name}")
print("\nClassification Report:\n", classification_report(y_test, best_model.predict(X_test)))

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
results_df.to_csv(REPORT_PATH, index=False)
joblib.dump(best_model, MODEL_PATH)
print("Comparison saved to:", REPORT_PATH)
print("Best model saved to:", MODEL_PATH)