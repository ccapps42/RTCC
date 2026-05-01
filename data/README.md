# Data

Training data is NOT stored here. It lives at:

    K:\projects\OpenHobbs\data\hf_cache

Do NOT re-download. The loader reads directly from that cache directory.

## Validation set

`validation/val.parquet` — held-out validation set carved by `scripts/prepare_validation.py`.
Written ONCE before any training begins. Never used for training.
See `validation/README.md` for the exact split parameters and hash.
