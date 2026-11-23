"""BiLSTM baseline. Uses a train-only vocabulary and validation early stopping."""
import argparse, json, re, time
from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from src.config import DATA_PROCESSED, MODELS, REPORTS, SEED, LABEL2ID, ID2LABEL

TOKEN_RE = re.compile(r"[a-z0-9']+|[!?.,]")
def tokenize(s): return TOKEN_RE.findall(str(s).lower())

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=160):
        self.texts, self.labels, self.vocab, self.max_len = list(texts), list(labels), vocab, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        ids = [self.vocab.get(t, 1) for t in tokenize(self.texts[i])[:self.max_len]]
        ids = ids + [0] * (self.max_len-len(ids))
        return torch.tensor(ids), torch.tensor(LABEL2ID[self.labels[i]])

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden=96, dropout=0.35):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden*2, 3)
    def forward(self, x):
        emb = self.embedding(x)
        out, (h, _) = self.lstm(emb)
        pooled = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(self.dropout(pooled))

def run_epoch(model, loader, criterion, optimizer, device):
    training = optimizer is not None
    model.train(training); losses=[]; ys=[]; ps=[]
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        if training: optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits=model(x); loss=criterion(logits,y)
            if training: loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        losses.append(loss.item()); ys.extend(y.cpu().tolist()); ps.extend(logits.argmax(1).cpu().tolist())
    labels=list(range(3))
    return float(np.mean(losses)), accuracy_score(ys,ps), f1_score(ys,ps,labels=labels,average="macro"), ys, ps

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=8); ap.add_argument("--batch-size",type=int,default=64)
    ap.add_argument("--max-vocab",type=int,default=30000); ap.add_argument("--max-len",type=int,default=160)
    ap.add_argument("--emb-dim",type=int,default=128); ap.add_argument("--dropout",type=float,default=0.35)
    args=ap.parse_args()
    torch.manual_seed(SEED); np.random.seed(SEED)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr=pd.read_csv(DATA_PROCESSED/"train.csv").fillna(""); va=pd.read_csv(DATA_PROCESSED/"validation.csv").fillna("")
    counts=Counter(t for s in tr.text for t in tokenize(s))
    vocab={"<pad>":0,"<unk>":1}
    vocab.update({w:i+2 for i,(w,_) in enumerate(counts.most_common(args.max_vocab-2))})
    train_ds=ReviewDataset(tr.text,tr.label,vocab,args.max_len); val_ds=ReviewDataset(va.text,va.label,vocab,args.max_len)
    train_dl=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True)
    val_dl=DataLoader(val_ds,batch_size=args.batch_size)
    model=BiLSTM(len(vocab),args.emb_dim,96,args.dropout).to(device)
    weights=torch.tensor([1.0/max(1,(tr.label==name).sum()) for name in ["Negative","Neutral","Positive"]],dtype=torch.float32)
    weights=weights/weights.sum()*3
    criterion=nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best=-1; patience=2; stale=0; history=[]
    for epoch in range(1,args.epochs+1):
        tl,ta,tf,_,_=run_epoch(model,train_dl,criterion,optimizer,device)
        vl,va_acc,vf,_,_=run_epoch(model,val_dl,criterion,None,device)
        row={"epoch":epoch,"train_loss":tl,"train_accuracy":ta,"train_macro_f1":tf,"val_loss":vl,"val_accuracy":va_acc,"val_macro_f1":vf}
        history.append(row); print(row)
        if vf>best:
            best=vf; stale=0
            torch.save({"state_dict":model.state_dict(),"vocab":vocab,"config":vars(args)},MODELS/"bilstm.pt")
        else:
            stale+=1
            if stale>=patience: print("Early stopping"); break
    pd.DataFrame(history).to_csv(REPORTS/"bilstm_history.csv",index=False)
    print("Device:",device,"Best validation macro-F1:",best)

if __name__=="__main__": main()
