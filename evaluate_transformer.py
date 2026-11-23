"""Evaluate the fine-tuned DistilBERT on the held-out test partition."""
import pandas as pd, numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns
from src.config import DATA_PROCESSED, MODELS, REPORTS, LABEL2ID, LABELS

def main():
    path=MODELS/"distilbert_sentiment"
    test=pd.read_csv(DATA_PROCESSED/"test.csv").fillna("")
    test["labels"]=test.label.map(LABEL2ID)
    tokenizer=AutoTokenizer.from_pretrained(str(path))
    ds=Dataset.from_pandas(test[["text","labels"]],preserve_index=False)
    ds=ds.map(lambda b:tokenizer(b["text"],truncation=True,max_length=192),batched=True)
    model=AutoModelForSequenceClassification.from_pretrained(str(path))
    args=TrainingArguments(output_dir=str(MODELS/"transformer_test_eval"),per_device_eval_batch_size=16,report_to="none")
    trainer=Trainer(model=model,args=args,tokenizer=tokenizer,data_collator=DataCollatorWithPadding(tokenizer))
    pred=trainer.predict(ds).predictions.argmax(1)
    print(classification_report(test.label,pred,labels=LABELS,zero_division=0))
    cm=confusion_matrix(test.label,pred,labels=LABELS)
    sns.heatmap(cm,annot=True,fmt="d",xticklabels=LABELS,yticklabels=LABELS,cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
    plt.savefig(REPORTS/"distilbert_test_confusion_matrix.png",dpi=160); plt.close()
if __name__=="__main__": main()
