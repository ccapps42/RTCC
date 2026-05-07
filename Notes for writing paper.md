# Notes for Writing the Paper

---

## CART: Non-causal cross-attention data leakage bug (found and fixed)

The cross-attention in CART's recurrent core block was initially set to `is_causal=False`, meaning each hidden state position `h[t]` could attend to all positions in the prelude output `e`, including future positions `e[t+1]`, `e[t+2]`, etc.

**Why this is a problem:** `e` is built from the same token sequence the model is trying to predict. `e[t+1]` was constructed using token `t+1` as input. So `h[t]` — whose job is to predict token `t+1` — had indirect access to that token through `e[t+1]`. That's the model reading its own answer.

**How it manifested:** With a shallow prelude (P=2), `e` doesn't encode tokens richly enough for the model to easily exploit this. With P=3, after ~1,750 training steps the model figured out how to extract next-token information from `e` via the non-causal attention, and the loss collapsed to near zero.

**The fix:** Changed `is_causal=False` to `is_causal=True` in `MLACrossAttention`. Now `h[t]` can only attend to `e[0..t]` — the same causal constraint that applies everywhere else in the model.

**RTCC is not affected:** RTCC replaced cross-attention with the LTI point-wise injection (`b_param * e`), which operates independently at each position. `h[t]` only ever sees `e[t]` — no cross-position mixing is possible by construction.

---

## Related work: recurrent depth papers (2024–2026)

Search conducted 2026-05-07. Field is active — several papers dropped April 2026.

### Must cite / directly competitive

**Parcae: Scaling Laws For Stable Looped Language Models**
arXiv 2604.12946 — Prairie, Novack, Berg-Kirkpatrick, Fu
Most directly competitive. Establishes the first scaling laws for looped transformers. Key finding: 770M Parcae matches quality of 1.3B standard transformer on same data. Optimal recurrence scales as C^0.40, tokens as C^0.78. RTCC's differentiation: spatial ToroidalMoE structure is orthogonal to Parcae's scaling law focus — Parcae is about when to loop, RTCC is about what the loop body does.

**Hyperloop Transformers**
arXiv 2604.21254 — Zeitoun, Torroba-Hennigen, Kim
Already in component attribution (hyper-connections). Shows ~50% parameter reduction vs depth-matched standard transformers. Their efficiency result is consistent with RTCC's efficiency story and can be cited as supporting context.

**The Recurrent Transformer: Greater Effective Depth and Efficient Decoding**
arXiv 2604.21215 — Oncescu et al.
Listed in sweep plan attribution as "OpenMythos/Claude Mythos" for LTI injection. Verify this arXiv ID matches the intended citation before submission. Core claim: reduced KV cache memory and inference latency via layer-wise recurrence. Tiling algorithm reduces HBM traffic from Θ(N²) to Θ(N log N).

### Useful for framing / contrast

**Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers**
arXiv 2604.07822 — Kohli et al.
Found "overthinking" degradation at excessive recurrence depth. RTCC's LTI sigmoid gating (spectral radius < 1) is specifically designed to prevent instability at high R — make this contrast explicit. Also: their "three-stage grokking" finding (memorization → in-distribution → systematic generalization) is useful framing for what recurrent depth buys.

**LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation**
arXiv 2602.11451 — Jeddi, Ciccone, Taati
Dynamic depth adjustment via shortcut-consistency training. Shorter loops yield informative representations, longer loops refine. Relevant framing for RTCC's variable-R inference behavior.

**Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence**
arXiv 2511.07384 — McLeish, Li, Kirchenbauer et al. (includes Geiping)
Shows recurrent depth can be added post-hoc to pretrained models. RTCC trains from scratch with depth recurrence baked in — worth noting as a design choice contrast. From the same group as Huginn.

### Lower priority but track

**Thinking Deeper, Not Longer: Depth-Recurrent Transformers for Compositional Generalization**
arXiv 2603.21676 — Hung-Hsuan Chen
Decoupling computational depth from parameter count. Standard result, useful as supporting citation.

**RingFormer: Rethinking Recurrent Transformer with Adaptive Level Signals**
arXiv 2502.13181 — Heo et al.
Single layer processed circularly with adaptive level signals. Interesting but distant from RTCC's approach.

**Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer**
arXiv 2507.02199 — Lu et al.
Negative result: limited evidence of interpretable latent CoT in recurrent-depth models. Marginal gains from added recurrence underperform explicit CoT. Worth being aware of for reviewer questions.

### Missing / need to find

- **Huginn** (Geiping et al., 2025) — primary structural inspiration, Prelude/Core/Coda design. Did not appear in search results, may not have a public arXiv ID yet. Locate before submission.

---

## Parameter counts and leverage

> **Note:** Numbers below are pre-run estimates from model instantiation. Update this section once actual paper runs complete.

RTCC uses weight sharing in the recurrent core (same `RTCCBlock` executed R times). Effective parameters = total + core_block × (R − 1).

| Config | Total params | Effective params | Leverage | Core block |
|--------|-------------|-----------------|----------|------------|
| RTCC d=768 R=8 | 74.4M | 92.2M | 1.24x | 2.5M |

CART comparison (for context):

| Config | Total params | Effective params | Leverage |
|--------|-------------|-----------------|----------|
| CART d=768 R=8 P=6 | 75.3M | 116.6M | 1.55x |
| CART d=1024 R=8 P=6 | 125.1M | 200.3M | 1.60x |

RTCC's lower leverage (1.24x vs 1.55x) reflects that the ToroidalMoE core block is more parameter-efficient (2.5M) than CART's SwiGLU FFN core (5.9M). The recurrent core does more computation per parameter — the flip side is less amplification from weight sharing. Worth addressing in the paper.

RTCC has no 1024-dim config — the toroidal grid must equal `model_dim` exactly (32×24=768), so a 1024-dim variant would require a new grid layout (e.g. 32×32) and is not currently planned.
