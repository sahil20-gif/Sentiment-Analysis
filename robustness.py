"""Run qualitative robustness probes and quantitative slices for the saved TF-IDF model."""
import argparse
import pandas as pd
import joblib
from sklearn.metrics import f1_score, accuracy_score
from src.config import DATA_PROCESSED, MODELS, REPORTS, LABELS

PROBES = [
    ("negation_positive", "This product is good.", "Positive"),
    ("negation_negative", "This product is not good.", "Negative"),
    ("negation_negative", "The device works.", "Positive"),
    ("negation_negative", "The device does not work.", "Negative"),
    ("mixed_sentiment", "Great product quality but terrible delivery.", "Neutral"),
    ("informal", "looove it!!! works gr8 :)", "Positive"),
    ("misspelling", "The qualitty is exellent and it works perfctly.", "Positive"),
    ("neutral", "The package arrived on Tuesday.", "Neutral"),
    ("negative", "It broke after two days and customer support was unhelpful.", "Negative"),
]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="best_tfidf_model.joblib")
    args=ap.parse_args()
    model=joblib.load(MODELS/args.model)
    probes=pd.DataFrame(PROBES,columns=["experiment","text","expected"])
    probes["predicted"]=model.predict(probes.text)
    probes["correct"]=probes.expected==probes.predicted
    probes.to_csv(REPORTS/"robustness_probes.csv",index=False)
    print("Qualitative probe results (expected labels are illustrative, not gold annotations):")
    print(probes.to_string(index=False))

    test=pd.read_csv(DATA_PROCESSED/"test.csv").fillna("")
    test["prediction"]=model.predict(test.text)
    test["length_words"]=test.text.str.split().str.len()
    test["length_bucket"]=pd.cut(test.length_words, bins=[-1,10,40,10**9], labels=["short","medium","long"])
    slices=[]
    for bucket,part in test.groupby("length_bucket",observed=True):
        slices.append({"slice":"length","value":str(bucket),"n":len(part),
                       "accuracy":accuracy_score(part.label,part.prediction),
                       "macro_f1":f1_score(part.label,part.prediction,labels=LABELS,average="macro",zero_division=0)})
    for category,part in test.groupby("product_category"):
        if len(part)>=30:
            slices.append({"slice":"category","value":str(category),"n":len(part),
                           "accuracy":accuracy_score(part.label,part.prediction),
                           "macro_f1":f1_score(part.label,part.prediction,labels=LABELS,average="macro",zero_division=0)})
    pd.DataFrame(slices).to_csv(REPORTS/"robustness_slices.csv",index=False)
    print("\\nSaved robustness_probes.csv and robustness_slices.csv")
if __name__=="__main__": main()
