"""Download, normalize, audit, and group-split Amazon product reviews."""
import argparse
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import GroupShuffleSplit
from src.config import DATA_PROCESSED, SEED

DATASET_ID = "Kingpandaaaa/amazon_reviews_multi"

def stars_to_sentiment(stars):
    stars = int(stars)
    if stars <= 2:
        return "Negative"
    if stars == 3:
        return "Neutral"
    return "Positive"

def load_amazon_reviews(max_rows=60000, seed=SEED):
    """Load English reviews. max_rows=None loads the complete English corpus."""
    ds = load_dataset(DATASET_ID, "en")
    frames = []
    for split_name in ("train", "validation", "test"):
        frame = ds[split_name].to_pandas()
        frame["source_split"] = split_name
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    # Deterministic sampling before cleaning keeps laptop experiments manageable.
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)
    df = df.rename(columns={"review_body": "text"})
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df = df.dropna(subset=["stars"])
    df = df[df["stars"].between(1, 5)]
    df = df[df["text"].str.len() > 0].copy()
    df["label"] = df["stars"].astype(int).map(stars_to_sentiment)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df[["text", "label", "stars", "product_id", "product_category", "source_split"]]

def group_train_val_test_split(df, seed=SEED):
    """Split by product_id to reduce product-specific leakage across partitions."""
    groups = df["product_id"].fillna(df.index.astype(str)).astype(str)
    first = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, temp_idx = next(first.split(df, df["label"], groups))
    train = df.iloc[train_idx].copy()
    temp = df.iloc[temp_idx].copy()
    temp_groups = temp["product_id"].fillna(temp.index.astype(str)).astype(str)
    second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(second.split(temp, temp["label"], temp_groups))
    return train.reset_index(drop=True), temp.iloc[val_idx].reset_index(drop=True), temp.iloc[test_idx].reset_index(drop=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=60000, help="0 means load all available English rows")
    args = parser.parse_args()
    df = load_amazon_reviews(max_rows=args.max_rows or None)
    train, val, test = group_train_val_test_split(df)
    for name, part in [("train", train), ("validation", val), ("test", test)]:
        part.to_csv(DATA_PROCESSED / f"{name}.csv", index=False)
    audit = {
        "dataset": DATASET_ID, "rows_after_cleaning": len(df),
        "train_rows": len(train), "validation_rows": len(val), "test_rows": len(test),
        "label_distribution": df["label"].value_counts().to_dict(),
        "product_overlap_train_validation": len(set(train.product_id) & set(val.product_id)),
        "product_overlap_train_test": len(set(train.product_id) & set(test.product_id)),
        "note": "Labels are derived from star ratings: 1-2 Negative, 3 Neutral, 4-5 Positive."
    }
    pd.Series(audit).to_json(DATA_PROCESSED / "dataset_audit.json", indent=2)
    print(pd.Series(audit).to_string())
    print("Saved CSV files to", DATA_PROCESSED)

if __name__ == "__main__":
    main()
