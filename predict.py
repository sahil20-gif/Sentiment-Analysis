import argparse, joblib
from pathlib import Path
from src.config import MODELS

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("review",nargs="?",default="The product works perfectly and the quality is excellent.")
    ap.add_argument("--model",default="best_tfidf_model.joblib")
    args=ap.parse_args()
    path=MODELS/args.model
    if not path.exists(): raise SystemExit(f"Model not found: {path}. Train the ML model first.")
    model=joblib.load(path)
    print({"review":args.review,"sentiment":model.predict([args.review])[0]})
    if hasattr(model,"predict_proba"):
        probs=model.predict_proba([args.review])[0]
        print("Probabilities:",dict(zip(model.classes_,probs.round(4))))

if __name__=="__main__": main()
