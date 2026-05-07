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
arXiv 2604.12946 — Prairie, Novack, Berg-Kirkpatrick, Fu (UC San Diego / Together AI)

Most directly competitive. Read in full 2026-05-07.

*Stability mechanism:* Frames the looped residual stream as a dynamical system: `h_{t+1} = A_bar * h_t + B_bar * e + R_bar(h_t, e)`. Stability requires spectral radius ρ(A_bar) < 1. Their fix: restrict A to negative diagonal, discretize via ZOH: `A_bar = exp(Delta * A)`, which is elementwise < 1 by construction. **This is the same stability requirement as RTCC's LTI** — RTCC uses `sigmoid(a_param)` as a different parameterization of the same guarantee. Cite Parcae and note both approaches solve the same dynamical systems problem. Their key empirical finding: all divergent runs had ρ ≥ 1; all convergent runs had ρ < 1. Standard addition-based injection gives exactly ρ = 1 (marginally stable), which is why explicit gating is needed.

*Quality:* 770M Parcae matches 1.3B standard transformer. 1.3B Parcae at T=8 loops scores 28.44 on Core benchmark vs 25.45 for same-size transformer.

*Scaling laws:*
- Training: L̂(μ_rec, D) = E + X·N(μ_rec)^(−x) + Y·D^(−y); exponents γ_μ ≈ 0.40, γ_D ≈ 0.78
- Looping is a third scaling axis alongside model size and data
- Inference: L(T) = L_∞ + Z·exp(−z·T); saturating exponential, ceiling set by training depth

*Differentiation from RTCC:* Parcae asks "when and how much to loop" (scaling law focus). RTCC asks "what the loop body does spatially" (ToroidalMoE structure). Orthogonal contributions — Parcae's stability result actually *supports* RTCC's LTI design choice.

**Hyperloop Transformers**
arXiv 2604.21254 — Zeitoun, Torroba-Hennigen, Kim (MIT)

Read in full 2026-05-07. Already in component attribution (hyper-connections).

*Main result:* Hyperloop (135.7M params) outperforms depth-matched Transformer (238M params) in perplexity and downstream accuracy — ~50% parameter reduction with better quality. Holds after INT4 quantization.

*Hyperconnection mechanics:* Expands scalar residual stream to matrix-valued `T × n × C`. Three learned input-dependent projection matrices (H^pre, H^post, H^res) gate the per-loop update. Applied at the loop boundary only — once per loop iteration, not once per layer.

*n-value ablation* (the key question for RTCC):

| n | PPL |
|---|-----|
| 2 | 14.429 |
| 4 | 14.404 |
| 6 | 14.379 |
| 8 | 14.388 |
| 10 | 14.349 |

Diminishing returns above n=2. Total gain n=4→n=10 is only 0.055 PPL; non-monotone at n=8. Authors chose n=4 following the upstream mHC paper. **RTCC uses n=3, which sits between the two clearest data points and is well-justified** — the paper explicitly states diminishing returns above n=2, so n=3 is past the meaningful threshold without the parameter overhead of higher n. Cite: "diminishing returns above n=2 observed in Zeitoun et al.; we use n=3."

*Loop placement finding:* Hyper-connections once per loop (3 total for L=3) outperforms placement at every layer (12 total) — 14.40 vs 14.45 PPL — while being far cheaper. CART and RTCC apply once per loop iteration. This is correct.

*2604.21106 cross-reference:* Schwethelm et al. tested K=2 lanes (their terminology) and found φ=0.65. Hyperloop's ablation shows n=2 already captures most gain. So the φ=0.65 result likely holds at n=3 as well — possibly slightly higher.

**The Recurrent Transformer: Greater Effective Depth and Efficient Decoding**
arXiv 2604.21215 — Oncescu et al.
Listed in sweep plan attribution as "OpenMythos/Claude Mythos" for LTI injection. Verify this arXiv ID matches the intended citation before submission. Core claim: reduced KV cache memory and inference latency via layer-wise recurrence. Tiling algorithm reduces HBM traffic from Θ(N²) to Θ(N log N).

**How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models**
arXiv 2604.21106 — Schwethelm, Rückert, Kaissis (TU Munich / Imperial College)

Read in full 2026-05-07. 116 pretraining runs, r ∈ {1, 2, 4, 8}, ~50× compute range.

*Core metric — recurrence-equivalence exponent φ:* Derives a joint scaling law:
`L = E + A(N_once + r^φ · N_rec)^(−α) + B · D^(−β)`
where N_once = prelude+coda params, N_rec = recurrent block params. φ quantifies how much capacity each additional loop recovers relative to an unshared block. φ = 1 means a loop is as good as a new layer; φ = 0 means loops add nothing.

*Key results:*
- φ = 0.459 (95% CI: [0.41, 0.53]) for baseline looped transformer — each loop recovers ~half a real layer's capacity
- φ = 0.65 with hyperconnections — **RTCC uses hyperconnections (n=3), so this prediction applies directly**
- φ = 0.38 with TBPTT (truncated backprop through time) — worse than baseline; avoid
- R² = 0.9972 for joint law vs R² = 0.9552 if φ forced to 1

*Practical finding:* At r=4, a 410M looped model performs like a 580M non-looped model but costs training compute of a 1B non-looped model. Looped models trail non-looped by 0.03–0.12 nats depending on r.

*Iso-depth vs iso-parameter:* Their design fixes total effective layers = 20 (iso-depth), varying how many are looped. Parcae uses iso-parameter. RTCC is closest to iso-depth (P=6 prelude + R loops + coda=1, fixed structure). Clarify framing in paper.

*Implications for RTCC:*
- φ = 0.65 with hyperconnections is a concrete prediction your sweep will test. If RTCC's recurrent ToroidalMoE body matches or exceeds this, that is a publishable result.
- TBPTT reduces φ — RTCC backprops through all R loops, which is the right choice. Cite this as justification.
- The φ metric could be computed from RTCC sweep results and reported as a contribution.
- Looped models benefit *more* from data scaling than non-looped (their data exponent β/(α+β) = 0.61–0.67 vs 0.52 for non-looped). This supports training longer (500M tokens) rather than wider.

*Differentiation from RTCC:* They study the loop body as a standard transformer block. RTCC's contribution is the spatial structure of the loop body (ToroidalMoE). Their φ metric is a tool RTCC can adopt and extend.

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
