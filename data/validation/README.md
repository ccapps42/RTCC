# Validation Set — NO LONGER IN USE

`val.parquet` was created early in the project as a FineWeb-only validation set
tokenized with GPT-2. **The trainer no longer reads it.**

We switched to CART's three per-source validation bins to make the RTCC-vs-CART
perplexity comparison exact:

  - `K:\projects\Model_Paper_1\data\val\tinystories_val.bin`
  - `K:\projects\Model_Paper_1\data\val\wikipedia_val.bin`
  - `K:\projects\Model_Paper_1\data\val\fineweb_edu_val.bin`

These are Llama-2 32K tokenized (matching CART's training bins) and evaluated
50 batches per source, exactly matching CART's eval-time budget. See
`shared/eval/perplexity.py::evaluate_perplexity_bin`.

`val.parquet` is preserved on disk for historical reproducibility but referenced
by nothing in the trainer or eval code paths.

## Original parameters (historical)

- Dataset: HuggingFaceFW/fineweb (sample-10BT)
- Skip: 9,000,000 documents
- Documents used: 50,000
- Sequences: 68,409
- seq_len: 512
- Seed: 42
- Tokenizer: gpt2 (vocab 50,257)
- SHA256: 87a354e71933c431a26c76d78d8df6768f1137c2b9f1116b57458cb13d2e5ff0
- Created: 2026-05-01T18:32:10.826734+00:00
