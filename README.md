# Intelligent Text Classification & Sentiment Analysis

An end-to-end, reproducible NLP project that compares TF-IDF machine learning, a BiLSTM neural network, and fine-tuned DistilBERT for three-class Amazon product-review sentiment classification.

## 1. Dataset and label policy

Dataset: [Kingpandaaaa/amazon_reviews_multi on Hugging Face](https://huggingface.co/datasets/Kingpandaaaa/amazon_reviews_multi), a data-files mirror of the Amazon Reviews Multi corpus. The English configuration contains product review text (`review_body`), star ratings (`stars`), product identifiers and product categories.

The original corpus has separate train/validation/test source partitions. This project combines the available English partitions, removes exact duplicate review text, and makes a deterministic **product-group split** so reviews for the same product are not deliberately distributed across our new train/validation/test sets.

Ground-truth mapping (documented weak supervision from star ratings):
- 1–2 stars → Negative
- 3 stars → Neutral
- 4–5 stars → Positive

This is an operational mapping, not a claim that star ratings perfectly represent linguistic sentiment. Mixed sentiment, rating bias, review quality, and category imbalance are limitations. Dataset source/licensing terms should be reviewed before redistribution. The raw dataset is downloaded by `datasets`; it is not committed to this repository.

## 2. Project layout

```text
sentiment-analysis-project/
├── data/{raw,processed}/
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_traditional_ml.ipynb
│   ├── 03_deep_learning.ipynb
│   └── 04_transformer.ipynb
├── src/
│   ├── config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── train_ml.py
│   ├── train_dl.py
│   ├── train_transformer.py
│   ├── evaluate.py
│   └── predict.py
├── models/
├── reports/
├── requirements.txt
└── README.md
```

## 3. Setup (Windows PowerShell)

```powershell
cd sentiment-analysis-project
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` is unavailable, install Python 3.11 first or use your existing Python environment. For PyTorch, use the official selector at https://pytorch.org/get-started/locally/ to choose the correct CPU/CUDA wheel for your machine.

## 4. Run the project

### Step A — download, clean, audit, split

```powershell
python -m src.data --max-rows 60000
```

Use `--max-rows 0` to load all English rows (requires more RAM and time). The default 60,000-row cap is intended for a laptop-sized first experiment. Outputs are saved in `data/processed/`. The split is grouped by `product_id`; inspect `dataset_audit.json` to verify overlap counts and class balance.

### Step B — traditional ML experiments

```powershell
python -m src.train_ml
```

Compares unigram TF-IDF + Logistic Regression, unigram/bigram TF-IDF + Logistic Regression, Linear SVM, and Multinomial Naive Bayes. Selects the best **validation macro-F1** model and saves it to `models/best_tfidf_model.joblib`.

### Step C — BiLSTM

```powershell
python -m src.train_dl --epochs 8 --batch-size 64
```

The vocabulary is built from training text only. Padding/truncation is applied to sequences. The model uses class-weighted cross-entropy, AdamW, gradient clipping, dropout, and validation macro-F1 early stopping. Try `--emb-dim 64`, `--emb-dim 200`, `--max-len 100`, or `--dropout 0.5` as controlled experiments. Record each run's configuration and validation score.

### Step D — DistilBERT fine-tuning

```powershell
python -m src.train_transformer --epochs 2 --batch-size 8
```

The default fine-tuning subset is 12,000 training and 2,500 validation examples to keep compute manageable. Increase those caps if GPU/RAM allows. CPU training may be very slow. Saved model/tokenizer: `models/distilbert_sentiment/`.

### Step E — final test evaluation for selected ML model

```powershell
python -m src.evaluate
```

This evaluates the saved ML model on the untouched test partition, writes a classification report, confusion matrix, and misclassified examples under `reports/`. Do not use test results to tune parameters or select models. For a fair final comparison, evaluate each model on exactly the same test rows after all choices have been finalized.

For final DL/transformer test evaluation (only after model decisions are frozen):

```powershell
python -m src.evaluate_dl
python -m src.evaluate_transformer
```

### Step G — robustness and prediction

```powershell
python -m src.robustness
```

This writes qualitative probes (whose expected labels are illustrative, not gold labels) and quantitative test slices by review length and product category.

### Step H — predict a new review

```powershell
python -m src.predict "The delivery was late but the product quality is excellent."
```

## 5. Notebook workflow

Open the notebooks in Jupyter:

```powershell
jupyter notebook
```

Run notebooks in order. The notebooks are experiment guides and call the same reusable project code. Execute the training commands from a notebook cell using `%run` or from the terminal.

## 6. Evaluation strategy

Primary selection metric: **macro F1**, because it gives equal weight to Negative, Neutral, and Positive classes. Also report accuracy, per-class precision/recall/F1, and confusion matrices. Keep train/validation/test partitions separate:
- Train: fit vocabulary/vectorizer/model weights.
- Validation: compare architectures and tune hyperparameters.
- Test: one final unbiased evaluation after model selection.

Do not claim scores until the code has been run. Add the actual CSV/JSON outputs and plots to your report after training.

## 7. Required robustness experiments

Run and document at least three:
1. **Negation pairs:** compare “good” / “not good”, “works” / “does not work”.
2. **Mixed sentiment:** “great quality but terrible delivery”.
3. **Noise:** misspellings, repeated punctuation, URLs, and informal wording.
4. **Length slices:** evaluate short, medium, and long test reviews separately.
5. **Unseen vocabulary:** report OOV rate for the BiLSTM and compare model behavior.
6. **Product-category slices:** report macro-F1 by `product_category` where each slice has enough examples.

A small hand-written probe set is a qualitative stress test, not a replacement for labelled test data.

## 8. Preprocessing decisions

The classical pipeline uses light cleaning and TF-IDF word unigrams/bigrams. Negation and punctuation are retained because “good” and “not good” differ. Stop-word removal can remove useful negators (e.g. *not*, *never*), so it is not applied by default. Stemming/lemmatization can reduce vocabulary sparsity but may distort sentiment-bearing forms; test them as ablations rather than assuming they help.

DistilBERT uses its own tokenizer and raw-ish text. Do not apply stemming, stop-word removal, or aggressive punctuation stripping before a pretrained transformer.

## 9. Limitations and honest reporting

- Star-to-sentiment conversion is weak supervision; borderline and mixed reviews can be noisy.
- Dataset reviews are from a particular time period, marketplace, language, and category mix.
- Product-group split reduces product leakage but can change class balance; inspect distributions.
- BiLSTM and transformer results depend on hardware, sample size, seeds, and hyperparameters.
- A model trained on Amazon reviews may not transfer to tweets, support tickets, or other domains without evaluation/adaptation.
- This repository is a training/evaluation project, not a production moderation or customer-impact system.

## 10. Interview explanation

**Problem:** classify customer product reviews into negative, neutral, and positive sentiment.

**Approach:** derive transparent three-class labels from 1–5 star ratings; audit and deduplicate data; split by product ID; establish TF-IDF baselines; compare linear classifiers; train a BiLSTM; fine-tune DistilBERT; select with validation macro-F1; evaluate once on held-out test data; inspect failure cases and robustness.

**Why macro-F1?** A model can achieve high accuracy by favoring a frequent class. Macro-F1 treats each sentiment class equally.

**Why keep negation?** Removing “not” can reverse sentiment. Word bigrams can capture phrases such as “not good”.

**Why DistilBERT?** Contextual representations help with word meaning in context and long-range relationships, but require more memory and compute than sparse linear models.

## 11. Reproducibility checklist

- [ ] Record Python/package versions and hardware.
- [ ] Record random seed and exact command/config for every run.
- [ ] Save validation comparison CSV.
- [ ] Do not tune on test data.
- [ ] Evaluate all final candidates on the same test rows.
- [ ] Inspect at least 10 wrong predictions and explain them.
- [ ] Add actual results and limitations; never invent metrics.
- [ ] Add your own GitHub repository URL after pushing this folder.
