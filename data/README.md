# Data

Training data is NOT stored here. RTCC reads CART's pre-tokenized bins directly:

    K:\projects\Model_Paper_1\data\stage2\stage2_train.bin   — 1B tokens, Llama-2 32K tokenizer
    K:\projects\Model_Paper_1\data\val\tinystories_val.bin   — 500k tokens
    K:\projects\Model_Paper_1\data\val\wikipedia_val.bin     — 500k tokens
    K:\projects\Model_Paper_1\data\val\fineweb_edu_val.bin   — 500k tokens

Do NOT re-tokenize or relocate these. The training and validation loaders read
directly from those paths (see `shared/data/loader.py` and `shared/eval/perplexity.py`).
The shared HF dataset cache lives at `K:\data\hf_cache\` per user CLAUDE.md.

## Validation set

`validation/val.parquet` exists from an early phase of the project but is **no longer
used**. We switched to CART's three per-source val bins for direct CART parity (same
50-batch eval budget, same tokenizer). See `validation/README.md`.
