# Validation Set

Created: 2026-05-01T18:32:10.826734+00:00
Script: scripts/prepare_validation.py

## Parameters
- Dataset: HuggingFaceFW/fineweb (sample-10BT)
- Split: train
- Skip: 9,000,000 documents (held-out region, never seen during training)
- Documents used: 50,000
- Sequences: 68,409
- seq_len: 512
- Seed: 42
- EOS token id: 50256
- Tokenizer: gpt2

## Integrity
SHA256: 87a354e71933c431a26c76d78d8df6768f1137c2b9f1116b57458cb13d2e5ff0

## IMPORTANT
This file must NEVER be used for training. It is the sole validation reference
for all architectures. Do not modify, re-pack, or regenerate unless you restart
all experiments from scratch.
