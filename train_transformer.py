"""Fine-tune DistilBERT. Run on GPU when possible; start with --max-rows dataset cap."""
import argparse, numpy as np, pandas as pd, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.config import DATA_PROCESSED, MODELS, LABELS, LABEL2ID, ID2LABEL, SEED

def metrics(eval_pred):
    logits, labels = eval_pred
    pred=np.argmax(logits,axis=1)
    p,r,f,_=precision_recall_fscore_support(labels,pred,average="macro",zero_division=0)
    return {"accuracy":accuracy_score(labels,pred),"macro_precision":p,"macro_recall":r,"macro_f1":f}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-name",default="distilbert-base-uncased")
    ap.add_argument("--epochs",type=float,default=2)
    ap.add_argument("--batch-size",type=int,default=8)
    ap.add_argument("--max-length",type=int,default=192)
    ap.add_argument("--sample-train",type=int,default=12000)
    ap.add_argument("--sample-val",type=int,default=2500)
    args=ap.parse_args()
    tr=pd.read_csv(DATA_PROCESSED/"train.csv").fillna(""); va=pd.read_csv(DATA_PROCESSED/"validation.csv").fillna("")
    if args.sample_train and len(tr)>args.sample_train: tr=tr.sample(args.sample_train,random_state=SEED)
    if args.sample_val and len(va)>args.sample_val: va=va.sample(args.sample_val,random_state=SEED)
    for frame in (tr,va): frame["labels"]=frame.label.map(LABEL2ID)
    tokenizer=AutoTokenizer.from_pretrained(args.model_name)
    def tokenize(batch): return tokenizer(batch["text"],truncation=True,max_length=args.max_length)
    train_ds=Dataset.from_pandas(tr[["text","labels"]],preserve_index=False).map(tokenize,batched=True)
    val_ds=Dataset.from_pandas(va[["text","labels"]],preserve_index=False).map(tokenize,batched=True)
    model=AutoModelForSequenceClassification.from_pretrained(args.model_name,num_labels=3,id2label=ID2LABEL,label2id=LABEL2ID)
    out=MODELS/"distilbert_sentiment"
    training_args=TrainingArguments(
        output_dir=str(MODELS/"distilbert_checkpoints"), learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size*2,
        num_train_epochs=args.epochs, weight_decay=0.01, eval_strategy="epoch",
        save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, logging_steps=100, report_to="none",
        fp16=torch.cuda.is_available(), seed=SEED
    )
    trainer=Trainer(model=model,args=training_args,train_dataset=train_ds,eval_dataset=val_ds,
                    tokenizer=tokenizer,data_collator=DataCollatorWithPadding(tokenizer),compute_metrics=metrics)
    trainer.train()
    trainer.save_model(str(out)); tokenizer.save_pretrained(str(out))
    print("Saved transformer to",out)
    print("Validation metrics:",trainer.evaluate())

if __name__=="__main__": main()
