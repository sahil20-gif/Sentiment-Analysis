import argparse, json, time
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import ParameterGrid
from src.preprocessing import clean_for_classical
from src.config import DATA_PROCESSED, MODELS, REPORTS, SEED, LABELS

def score(model, X, y):
    pred = model.predict(X)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=LABELS, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "macro_precision": p, "macro_recall": r, "macro_f1": f}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-features", type=int, default=50000)
    args = ap.parse_args()
    train = pd.read_csv(DATA_PROCESSED / "train.csv").fillna("")
    val = pd.read_csv(DATA_PROCESSED / "validation.csv").fillna("")
    Xtr = train.text.map(clean_for_classical); ytr = train.label
    Xv = val.text.map(clean_for_classical); yv = val.label
    experiments = []
    configs = [
        ("tfidf_uni_logreg", (1,1), LogisticRegression(max_iter=1200, class_weight="balanced", C=2.0)),
        ("tfidf_bi_logreg", (1,2), LogisticRegression(max_iter=1200, class_weight="balanced", C=2.0)),
        ("tfidf_bi_svm", (1,2), LinearSVC(class_weight="balanced", C=1.0)),
        ("tfidf_bi_nb", (1,2), MultinomialNB(alpha=0.5)),
    ]
    best = None
    for name, ngram, clf in configs:
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=ngram, max_features=args.max_features,
                                      min_df=2, max_df=0.98, sublinear_tf=True,
                                      preprocessor=clean_for_classical)),
            ("clf", clf)
        ])
        start = time.time()
        pipe.fit(train.text, ytr)
        metrics = score(pipe, val.text, yv)
        row = {"model": name, **metrics, "seconds": round(time.time()-start, 2),
               "vocabulary_size": len(pipe.named_steps["tfidf"].vocabulary_)}
        experiments.append(row)
        print(row)
        if best is None or metrics["macro_f1"] > best[0]:
            best = (metrics["macro_f1"], name, pipe)
    pd.DataFrame(experiments).to_csv(REPORTS / "traditional_model_comparison.csv", index=False)
    joblib.dump(best[2], MODELS / "best_tfidf_model.joblib")
    (REPORTS / "best_ml_model.json").write_text(json.dumps({"model": best[1], "validation_macro_f1": best[0]}, indent=2))
    print("Saved best validation model:", best[1])

if __name__ == "__main__":
    main()
