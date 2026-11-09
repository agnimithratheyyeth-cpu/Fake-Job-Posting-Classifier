
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import shap
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os

# 1. Load dataset
df = pd.read_csv("dataset/fake_job_postings.csv")


# 2. Select important columns
text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements"
]


# 3. Check missing values
print("Missing values before cleaning:")
print(df[text_columns].isnull().sum())


# 4. Replace missing values with empty strings
df[text_columns] = df[text_columns].fillna("")

print("\nMissing values after cleaning:")
print(df[text_columns].isnull().sum())


# 5. Combine text columns
df["combined_text"] = (
    df["title"] + " " +
    df["company_profile"] + " " +
    df["description"] + " " +
    df["requirements"]
)

print("\nCombined text:")
print(df["combined_text"].iloc[0])


# 6. Define input and target
x = df["combined_text"]
y = df["fraudulent"]

print("\nInput shape:", x.shape)
print("Target shape:", y.shape)


# 7. Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", x_train.shape)
print("Testing data:", x_test.shape)


# 8. Create TF-IDF vectorizer
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

x_train_tfidf = tfidf.fit_transform(x_train)
x_test_tfidf = tfidf.transform(x_test)
models={
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    ),
    "Naive Bayes":MultinomialNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42
    )
}
for name, classifier in models.items():
    classifier.fit(x_train_tfidf, y_train)
    predictions = classifier.predict(x_test_tfidf)
    print("\nModel:", name)
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
model=models["Logistic Regression"]
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["liblinear", "lbfgs"]
}
grid_search= GridSearchCV(
    LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    ),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)
grid_search.fit(x_train_tfidf, y_train)
best_model= grid_search.best_estimator_
model= best_model
print("\nBest Parameters:",grid_search.best_params_)
print("Best Cross-Validation F1 Score:",
grid_search.best_score_)
tuned_predictions = best_model.predict(x_test_tfidf)
y_prob=best_model.predict_proba(x_test_tfidf)[:,1]
roc_auc= roc_auc_score(y_test,y_prob)
pr_auc= average_precision_score(y_test, y_prob)
print("\nROC-AUC Score:", roc_auc)
print("Precision-Recall AUC:",pr_auc)
print("\nTuned Logistic Regression Results:")
print(classification_report(y_test, tuned_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, tuned_predictions))

print("\nTraining TF-IDF shape:", x_train_tfidf.shape)
print("Testing TF-IDF shape:", x_test_tfidf.shape)


# 9. Train Logistic Regression model
model= best_model

model.fit(x_train_tfidf, y_train)

print("\nModel training completed!")


# 10. Make predictions
y_pred = model.predict(x_test_tfidf)

print("Prediction completed!")
print("First 10 predictions:", y_pred[:10])


# 11. Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 12. Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# 13. Confusion matrix heatmap using Matplotlib
plt.figure(figsize=(6, 5))

plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar()

plt.xticks([0, 1], ["Genuine", "Fake"])
plt.yticks([0, 1], ["Genuine", "Fake"])

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - Fake Job Classifier")

for i in range(2):
    for j in range(2):
        plt.text(
            j, i, cm[i, j],
            ha="center",
            va="center",
            color="black",
            fontsize=12
        )

plt.tight_layout()
plt.savefig("Confusion_matrix.png")
plt.show()


# 14. SHAP Explainable AI
explainer = shap.LinearExplainer(
    model,
    x_train_tfidf
)

print("\nSHAP Explainer created successfully!")

shap_values = explainer(x_test_tfidf[:5])

print("SHAP values generated successfully!")
print("SHAP values shape:", shap_values.values.shape)


# 15. Get feature names
feature_names = tfidf.get_feature_names_out()

print("\nFirst 20 words:")
print(feature_names[:20])


# 16. Calculate SHAP feature importance for first test job
first_job_shap = shap_values.values[0]

shap_df = pd.DataFrame({
    "Word": feature_names,
    "SHAP Value": first_job_shap
})

shap_df["Importance"] = shap_df["SHAP Value"].abs()

shap_df = shap_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 important words:")
print(shap_df.head(10))


# 17. Select top 10 important words
top_words = shap_df.head(10).sort_values(
    by="SHAP Value"
)


# 18. SHAP feature importance bar chart
plt.figure(figsize=(9, 6))

plt.barh(
    top_words["Word"],
    top_words["SHAP Value"]
)

plt.axvline(x=0, color="black")

plt.title("SHAP Feature Importance")
plt.xlabel("SHAP Value")
plt.ylabel("Words")

plt.tight_layout()
plt.savefig("shap_feature_importance.png")
plt.show()

print("\nProject execution completed successfully!")
os.makedirs("model", exist_ok=True)
joblib.dump(model,"model/logistic_model.pkl")
joblib.dump(tfidf,"model/tfidf_vectorizer.pkl")
print("\nModel and vectorizer saved successfully!")
