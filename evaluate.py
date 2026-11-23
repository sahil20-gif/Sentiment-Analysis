import argparse, json
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from src.config import DATA_PROCESSED, MODELS, REPORTS, LABELS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="best_tfidf_model.joblib")
    args = ap.parse_args()
    test = pd.read_csv(DATA_PROCESSED / "test.csv").fillna("")
    model = joblib.load(MODELS / args.model)
    pred = model.predict(test.text)
    report = classification_report(test.label, pred, labels=LABELS, output_dict=True, zero_division=0)
    (REPORTS / "test_classification_report.json").write_text(json.dumps(report, indent=2))
    print(classification_report(test.label, pred, labels=LABELS, zero_division=0))
    cm = confusion_matrix(test.label, pred, labels=LABELS)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Test Confusion Matrix")
    plt.tight_layout(); plt.savefig(REPORTS / "confusion_matrix.png", dpi=160); plt.close()
    errors = test.assign(prediction=pred)
    errors = errors[errors.label != errors.prediction]
    errors.head(100).to_csv(REPORTS / "misclassified_reviews.csv", index=False)
    print("Saved confusion matrix and error examples to reports/")

if __name__ == "__main__":
    main()
