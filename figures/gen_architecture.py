"""
RTCC Full Architecture Block Diagram — top to bottom
Based on actual code structure.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

os.makedirs('figures', exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 28), facecolor='white')
ax.set_xlim(0, 11)
ax.set_ylim(0, 38)
ax.set_aspect('equal')
ax.axis('off')

# ── Colors ───────────────────────────────────────────────────────────────────
C_IO      = '#d0e8ff'
C_ATTN    = '#dceeff'
C_FFN     = '#dcf5e8'
C_HYPER   = '#ecdcf5'
C_LIE     = '#fde8d0'
C_EXPERT  = '#e8d5f5'
C_LTI     = '#fff0cc'
C_NORM    = '#ffe0e0'
C_ANCHOR  = '#fffacc'
C_LMH     = '#cce5cc'
EDGE      = '#444444'
ARROW     = '#333333'

CX  = 5.2    # center x
BW  = 6.4    # box width
BH  = 1.15   # standard box height
BHT = 1.55   # tall box height
GAP = 0.42   # gap between boxes
pad = 0.62   # dashed-box padding

def box(ax, x, y, w, h, label, color, sublabel=None, fs=11):
    b = FancyBboxPatch((x-w/2, y-h/2), w, h,
                        boxstyle="round,pad=0.0,rounding_size=0.18",
                        facecolor=color, edgecolor=EDGE, linewidth=1.2, zorder=2)
    ax.add_patch(b)
    ty = y + (0.18 if sublabel else 0)
    ax.text(x, ty, label, ha='center', va='center',
            fontsize=fs, fontweight='bold', color='#1a1a1a', zorder=3)
    if sublabel:
        ax.text(x, y-0.28, sublabel, ha='center', va='center',
                fontsize=9, color='#444444', style='italic', zorder=3)

def arr(ax, x, y1, y2):
    ax.annotate('', xy=(x, y2+0.02), xytext=(x, y1-0.02),
                arrowprops=dict(arrowstyle='->', color=ARROW, lw=1.4), zorder=3)

def section_box(ax, x, y_top, y_bot, w, color, edge, label, label_color):
    lx = x - w/2 - pad
    lw = w + 2*pad
    lh = y_top - y_bot
    r = FancyBboxPatch((lx, y_bot), lw, lh,
                        boxstyle="round,pad=0.0,rounding_size=0.3",
                        facecolor=color, edgecolor=edge,
                        linewidth=1.8, linestyle='--', zorder=0)
    ax.add_patch(r)
    ax.text(lx+0.22, y_bot+lh/2, label,
            ha='center', va='center', fontsize=12, fontweight='bold',
            color=label_color, rotation=90, zorder=1)

# ── Layout ───────────────────────────────────────────────────────────────────
y = 37.0

# 1. Input Tokens
box(ax, CX, y, BW, BH, 'Input Tokens', C_IO)
y -= BH/2 + GAP; arr(ax, CX, y+GAP-0.02, y+0.02); y -= BH/2

# ── PRELUDE ×4 layers ────────────────────────────────────────────────────────
prelude_top = y + BH/2 + GAP*0.35

box(ax, CX, y, BW, BH, 'Multi-head Latent Attention',  C_ATTN,
    sublabel='RMSNorm → MLA  (residual add)')
y -= BH/2 + GAP; arr(ax, CX, y+GAP-0.02, y+0.02); y -= BH/2

box(ax, CX, y, BW, BH, 'SwiGLU FFN', C_FFN,
    sublabel='RMSNorm → SwiGLU Feed-Forward  (residual add)')
prelude_bot = y - BH/2 - GAP*0.35

# Store anchor e — branch to the right
anchor_y = y
section_box(ax, CX, prelude_top, prelude_bot, BW, '#f0fff4', '#228833', '×4  Prelude', '#228833')

# Anchor e side node
anx = CX + BW/2 + pad + 1.2
any_ = anchor_y
ax.annotate('', xy=(anx-0.5, any_), xytext=(CX+BW/2+0.02, any_),
            arrowprops=dict(arrowstyle='->', color='#885500', lw=1.3, linestyle='dashed'), zorder=3)
eb = FancyBboxPatch((anx-0.5, any_-0.38), 1.0, 0.76,
                     boxstyle="round,pad=0.0,rounding_size=0.12",
                     facecolor=C_ANCHOR, edgecolor='#885500', linewidth=1.2, zorder=2)
ax.add_patch(eb)
ax.text(anx, any_, 'e\n(anchor)', ha='center', va='center',
        fontsize=9, fontweight='bold', color='#885500', zorder=3)

y -= BH/2 + GAP; arr(ax, CX, y+GAP-0.02, y+0.02); y -= BH/2

# ── RECURRENT CORE ×R ────────────────────────────────────────────────────────
loop_top = y + BH/2 + GAP*0.35

box(ax, CX, y, BW, BH, 'hyper.combine(buffer)',  C_HYPER,
    sublabel='Blend hyper-connection buffer states  →  h_input')
y -= BH/2 + GAP; arr(ax, CX, y+GAP-0.02, y+0.02); y -= BH/2

box(ax, CX, y, BW, BH, 'Loop Index Embedding (LIE)', C_LIE,
    sublabel='Sinusoidal loop-index encoding → projected to model_dim\n→ added to h_input')
y -= BH/2 + GAP; arr(ax, CX, y+GAP-0.02, y+0.02); y -= BH/2

box(ax, CX, y, BW, BH, 'Multi-head Latent Attention', C_ATTN,
    sublabel='RMSNorm → MLA self-attention  (residual add)')
y -= BH/2 + GAP; arr(ax, CX, y+GAP-0.02, y+0.02); y -= BH/2

box(ax, CX, y, BW, BHT, 'Toroidal Expert Block', C_EXPERT,
    sublabel='RMSNorm → circular (toroidal) pad → unfold → per-expert SwiGLU\n→ overlap-add fold  (residual add)')
y -= BHT/2 + GAP; arr(ax, CX, y+GAP-0.02, y+0.02); y -= BH/2

box(ax, CX, y, BW, BH, 'Linear Time-Invariant Update (LTI)', C_LTI,
    sublabel='h = sigmoid(A)·h_input + B·e + transformer_out')
# Dashed line from anchor e down to LTI
lti_y = y
ax.plot([anx, anx], [any_-0.38, lti_y+0.15], color='#885500', lw=1.2,
        linestyle='--', zorder=1)
ax.annotate('', xy=(CX+BW/2+pad+0.08, lti_y), xytext=(anx, lti_y),
            arrowprops=dict(arrowstyle='->', color='#885500', lw=1.2, linestyle='dashed'), zorder=3)

y -= BH/2 + GAP; arr(ax, CX, y+GAP-0.02, y+0.02); y -= BH/2

box(ax, CX, y, BW, BH, 'hyper.update_buffer(buffer, h)', C_HYPER,
    sublabel='Push h to front of ring buffer, drop oldest state')
loop_bot = y - BH/2 - GAP*0.35

section_box(ax, CX, loop_top, loop_bot, BW, '#f8f4ff', '#7755aa', '×R  Recurrent Core', '#7755aa')

# Loop back arrow
rx = CX + BW/2 + pad + 0.55
ry_bot = loop_bot + 0.18
ry_top = loop_top - 0.18
ax.annotate('', xy=(rx, ry_top), xytext=(rx, ry_bot),
            arrowprops=dict(arrowstyle='->', color='#7755aa', lw=1.8))
ax.plot([CX+BW/2+pad, rx], [ry_bot, ry_bot], color='#7755aa', lw=1.8)
ax.plot([CX+BW/2+pad, rx], [ry_top, ry_top], color='#7755aa', lw=1.8)
ax.text(rx, (ry_bot+ry_top)/2, '× R loops',
        ha='center', va='center', fontsize=12, fontweight='bold',
        color='#7755aa', rotation=90,
        bbox=dict(facecolor='white', edgecolor='none', alpha=1.0, pad=4))

y -= BH/2 + GAP; arr(ax, CX, y+GAP-0.02, y+0.02); y -= BH/2

# ── CODA ×1 layer ────────────────────────────────────────────────────────────
coda_top = y + BH/2 + GAP*0.35

box(ax, CX, y, BW, BH, 'Multi-head Latent Attention', C_ATTN,
    sublabel='RMSNorm → MLA  (residual add)')
y -= BH/2 + GAP; arr(ax, CX, y+GAP-0.02, y+0.02); y -= BH/2

box(ax, CX, y, BW, BH, 'SwiGLU FFN', C_FFN,
    sublabel='RMSNorm → SwiGLU Feed-Forward  (residual add)')
coda_bot = y - BH/2 - GAP*0.35

section_box(ax, CX, coda_top, coda_bot, BW, '#fff8f0', '#cc6600', '×1  Coda', '#cc6600')

y -= BH/2 + GAP; arr(ax, CX, y+GAP-0.02, y+0.02); y -= BH/2

# LM Head
box(ax, CX, y, BW, BH, 'Language Model Head', C_LMH,
    sublabel='RMSNorm → Linear  (weight-tied to embedding)')
y -= BH/2 + GAP; arr(ax, CX, y+GAP-0.02, y+0.02); y -= BH/2

# Logits
box(ax, CX, y, BW, BH, 'Logits  →  Next Token', C_IO)

# ── Title ────────────────────────────────────────────────────────────────────
ax.text(CX, 37.80, 'RTCC Architecture',
        ha='center', va='center', fontsize=20, fontweight='bold',
        fontfamily='monospace', color='#111111')
ax.text(CX, 37.35, 'Recurrent Toroidal Cortical Columns',
        ha='center', va='center', fontsize=12, color='#555555', style='italic')

plt.tight_layout(pad=0.3)
plt.savefig('figures/rtcc_architecture.pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('figures/rtcc_architecture.png', dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
print("Saved.")
