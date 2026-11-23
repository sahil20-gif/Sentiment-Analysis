from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts"

for path in (DATA_RAW, DATA_PROCESSED, MODELS, REPORTS, ARTIFACTS):
    path.mkdir(parents=True, exist_ok=True)

SEED = 42
LABELS = ["Negative", "Neutral", "Positive"]
LABEL2ID = {name: idx for idx, name in enumerate(LABELS)}
ID2LABEL = {idx: name for name, idx in LABEL2ID.items()}
