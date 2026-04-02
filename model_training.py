import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from xgboost import XGBClassifier

# 1️⃣ Load Dataset
data = pd.read_csv("diabetes.csv")

# 2️⃣ Data Cleaning (Fixed - No Chained Assignment)
columns_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in columns_to_fix:
    data[col] = data[col].replace(0, np.nan)
    data[col] = data[col].fillna(data[col].median())

# 3️⃣ Split Data
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4️⃣ Models
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42
)

xgb = XGBClassifier(
    eval_metric='logloss',
    random_state=42
)

rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb)],
    voting='soft'
)

ensemble.fit(X_train, y_train)

# 5️⃣ Evaluation
models = {
    "Random Forest": rf,
    "XGBoost": xgb,
    "Ensemble": ensemble
}

print("\n========= MODEL PERFORMANCE =========\n")

for name, model in models.items():
    y_pred = model.predict(X_test)
    print("------", name, "------")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print(classification_report(y_test, y_pred))

# 6️⃣ Cross Validation (Stratified)
print("\n========= 10-Fold Cross Validation =========\n")

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

best_model = None
best_score = 0

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=skf)
    mean_score = scores.mean()
    print(name, "10-Fold CV Accuracy:", round(mean_score, 4))
    
    if mean_score > best_score:
        best_score = mean_score
        best_model = model

# 7️⃣ ROC Curve
plt.figure(figsize=(8,6))

for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# 8️⃣ Save Best Model Automatically
pickle.dump(best_model, open("diabetes_model.pkl", "wb"))

print("\nBest Model Saved Successfully!")
print("Best CV Accuracy:", round(best_score, 4))
