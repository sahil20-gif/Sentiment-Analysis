"""Evaluate the saved BiLSTM once on the held-out test set."""
import pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns
from src.train_dl import BiLSTM, ReviewDataset
from src.config import DATA_PROCESSED, MODELS, REPORTS, ID2LABEL

def main():
    ckpt=torch.load(MODELS/"bilstm.pt",map_location="cpu",weights_only=False)
    cfg=ckpt["config"]; vocab=ckpt["vocab"]
    test=pd.read_csv(DATA_PROCESSED/"test.csv").fillna("")
    ds=ReviewDataset(test.text,test.label,vocab,cfg["max_len"])
    dl=DataLoader(ds,batch_size=cfg["batch_size"])
    model=BiLSTM(len(vocab),cfg["emb_dim"],96,cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    ys=[]; ps=[]
    with torch.no_grad():
        for x,y in dl:
            pred=model(x).argmax(1)
            ys.extend(y.tolist()); ps.extend(pred.tolist())
    names=[ID2LABEL[i] for i in range(3)]
    print(classification_report(ys,ps,target_names=names,zero_division=0))
    cm=confusion_matrix(ys,ps)
    sns.heatmap(cm,annot=True,fmt="d",xticklabels=names,yticklabels=names,cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
    plt.savefig(REPORTS/"bilstm_test_confusion_matrix.png",dpi=160); plt.close()
if __name__=="__main__": main()
