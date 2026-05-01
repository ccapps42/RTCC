"""ONE-TIME script: carve and save the validation set before any training begins.

Run this exactly once. The output val.parquet is sacred and must never be
used for training. A SHA256 hash is written alongside it to detect corruption.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VAL_DIR = PROJECT_ROOT / "data" / "validation"
VAL_PARQUET = VAL_DIR / "val.parquet"
VAL_README = VAL_DIR / "README.md"
VAL_HASH = VAL_DIR / "val.sha256"
HF_CACHE = Path("K:/projects/Model_Paper_1/data/hf_cache")

N_DOCS = 50_000
SEQ_LEN = 512
SEED = 42
EOS_TOKEN_ID = 2  # updated after tokenizer loads
DATASET_NAME = "HuggingFaceFW/fineweb"
DATASET_CONFIG = "sample-10BT"
# Use a held-out slice that training never touches
VAL_SPLIT = "train"
VAL_SKIP = 9_000_000  # skip 9M docs to avoid training overlap


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if VAL_PARQUET.exists():
        print(f"Validation set already exists at {VAL_PARQUET}")
        print("To re-create it, delete the file manually first.")
        return

    VAL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=str(HF_CACHE))
    eos_id = tokenizer.eos_token_id
    print(f"EOS token id: {eos_id}")

    print(f"Streaming {DATASET_NAME} ({DATASET_CONFIG}), skipping {VAL_SKIP:,} docs...")
    from datasets import load_dataset
    ds = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=VAL_SPLIT,
        streaming=True,
        cache_dir=str(HF_CACHE),
    )
    # Skip ahead to held-out region
    ds = ds.skip(VAL_SKIP)

    import random
    rng = random.Random(SEED)

    buf: list[int] = []
    inputs: list[list[int]] = []
    labels: list[list[int]] = []
    docs_seen = 0

    for example in ds:
        text = example.get("text", "")
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        buf.extend(ids)
        buf.append(eos_id)
        docs_seen += 1

        while len(buf) >= SEQ_LEN + 1:
            chunk = buf[:SEQ_LEN + 1]
            buf = buf[SEQ_LEN:]
            inputs.append(chunk[:SEQ_LEN])
            labels.append(chunk[1:SEQ_LEN + 1])

        if docs_seen >= N_DOCS:
            break

    print(f"Packed {len(inputs):,} validation sequences from {docs_seen:,} documents")

    import pandas as pd
    df = pd.DataFrame({"input_ids": inputs, "labels": labels})
    df.to_parquet(VAL_PARQUET, index=False)

    digest = sha256_file(VAL_PARQUET)
    VAL_HASH.write_text(digest)

    readme = f"""# Validation Set

Created: {datetime.now(timezone.utc).isoformat()}
Script: scripts/prepare_validation.py

## Parameters
- Dataset: {DATASET_NAME} ({DATASET_CONFIG})
- Split: {VAL_SPLIT}
- Skip: {VAL_SKIP:,} documents (held-out region, never seen during training)
- Documents used: {docs_seen:,}
- Sequences: {len(inputs):,}
- seq_len: {SEQ_LEN}
- Seed: {SEED}
- EOS token id: {eos_id}
- Tokenizer: gpt2

## Integrity
SHA256: {digest}

## IMPORTANT
This file must NEVER be used for training. It is the sole validation reference
for all architectures. Do not modify, re-pack, or regenerate unless you restart
all experiments from scratch.
"""
    VAL_README.write_text(readme)
    print(f"Validation set saved: {VAL_PARQUET}")
    print(f"SHA256: {digest}")
    print(f"README: {VAL_README}")


if __name__ == "__main__":
    main()
