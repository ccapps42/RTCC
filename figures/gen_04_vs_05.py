"""
Architecture 04 vs 05: Flat Grid Overlap vs RTCC
Full 32x24 grid, correct 10x10 patches, 6x6 private cores, 2-cell overlap borders.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
import os

os.makedirs('figures', exist_ok=True)

# ── True config ───────────────────────────────────────────────────────────────
GR, GC   = 32, 24    # grid rows, cols
PATCH    = 10        # patch size
STRIDE   = 8         # stride
OVERLAP  = 2         # = PATCH - STRIDE
PRIVATE  = 6         # = PATCH - 2*OVERLAP
ER, EC   = GR//STRIDE, GC//STRIDE   # 4×3 = 12 experts

CELL = 0.175   # figure units per grid cell

# Expert colors (12 experts, 4×3)
ECOLS = [
    ['#a8d8ea','#a8eaa8','#eaa8a8'],
    ['#b8c8f0','#b8f0c8','#f0b8b8'],
    ['#c8b8f0','#c8f0b8','#f0c8b8'],
    ['#d8a8e8','#d8e8a8','#e8d8a8'],
]
PRIVATE_DARKEN = 0.75   # darken factor for private core vs border
C_ZERO   = '#cccccc'
C_WRAP   = '#66bb6a'
C_BORDER = '#444444'

fig, axes = plt.subplots(1, 2, figsize=(16, 9), facecolor='white')
fig.patch.set_facecolor('white')

def darken(hex_color, factor=0.75):
    c = mpatches.FancyArrowPatch
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f'#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}'

def draw_panel(ax, flat_boundary):
    ax.set_facecolor('white')
    ax.set_aspect('equal')
    ax.axis('off')

    # Panel origin: grid starts at (ox, oy), grows right and down
    ox = OVERLAP * CELL + 0.3   # leave room for overhang on left
    oy = OVERLAP * CELL + 0.2   # leave room for overhang on top (y grows UP)

    grid_w = GC * CELL
    grid_h = GR * CELL

    # ── Draw grid background ──────────────────────────────────────────────────
    ax.add_patch(Rectangle((ox, oy), grid_w, grid_h,
                             facecolor='#f0f4ff', edgecolor='none', zorder=0))

    # Light grid lines
    for r in range(GR+1):
        y = oy + r*CELL
        ax.plot([ox, ox+grid_w], [y, y], color='#c0c8e0', lw=0.3, zorder=1)
    for c in range(GC+1):
        x = ox + c*CELL
        ax.plot([x, x], [oy, oy+grid_h], color='#c0c8e0', lw=0.3, zorder=1)

    # ── Draw expert patches ───────────────────────────────────────────────────
    for ei in range(ER):
        for ej in range(EC):
            col = ECOLS[ei][ej]
            pcol = darken(col, PRIVATE_DARKEN)

            # Expert (ei, ej): covers original grid rows [ei*S-OV : ei*S-OV+P]
            #                                       cols [ej*S-OV : ej*S-OV+P]
            r0 = ei * STRIDE - OVERLAP
            c0 = ej * STRIDE - OVERLAP

            # Draw each cell of the patch
            for pr in range(PATCH):
                for pc in range(PATCH):
                    gr = r0 + pr   # grid row (may be negative or ≥ GR)
                    gc = c0 + pc   # grid col (may be negative or ≥ GC)

                    # Is this cell inside the grid?
                    in_grid = (0 <= gr < GR) and (0 <= gc < GC)

                    # Is this cell in the private core?
                    in_private = (OVERLAP <= pr < PATCH-OVERLAP) and (OVERLAP <= pc < PATCH-OVERLAP)

                    cell_color = pcol if in_private else col
                    alpha = 0.80 if in_private else 0.55

                    if in_grid:
                        x = ox + gc * CELL
                        y = oy + gr * CELL
                        ax.add_patch(Rectangle((x+0.01, y+0.01), CELL-0.02, CELL-0.02,
                                                facecolor=cell_color, edgecolor='white',
                                                linewidth=0.4, alpha=alpha, zorder=2))
                    else:
                        # Outside the grid — boundary cell
                        # Where does it appear?
                        if flat_boundary:
                            # Zero padding — show gray hatched cell outside grid
                            x = ox + gc * CELL
                            y = oy + gr * CELL
                            ax.add_patch(Rectangle((x+0.01, y+0.01), CELL-0.02, CELL-0.02,
                                                    facecolor=C_ZERO, edgecolor='#999999',
                                                    linewidth=0.4, alpha=0.7,
                                                    hatch='////', zorder=2))
                        else:
                            # Toroidal — show as wrapped cell at outside position
                            # AND highlight the source cell inside the grid
                            x = ox + gc * CELL
                            y = oy + gr * CELL
                            ax.add_patch(Rectangle((x+0.01, y+0.01), CELL-0.02, CELL-0.02,
                                                    facecolor=C_WRAP, edgecolor='#2e7d32',
                                                    linewidth=0.6, alpha=0.80, zorder=3))

                            # Source cell (wrapped position inside grid)
                            src_r = gr % GR
                            src_c = gc % GC
                            sx = ox + src_c * CELL
                            sy = oy + src_r * CELL
                            ax.add_patch(Rectangle((sx+0.01, sy+0.01), CELL-0.02, CELL-0.02,
                                                    facecolor=C_WRAP, edgecolor='#2e7d32',
                                                    linewidth=0.6, alpha=0.45, zorder=2))

    # ── Grid border (draw on top) ─────────────────────────────────────────────
    ax.add_patch(Rectangle((ox, oy), grid_w, grid_h,
                             facecolor='none', edgecolor='#1a2a6f',
                             linewidth=2.0, zorder=5))

    # ── Annotate one expert's structure (bottom-left expert for clarity) ──────
    # Expert (0,0) — label private core and border
    ann_ei, ann_ej = 0, 0
    r0 = ann_ei * STRIDE - OVERLAP
    c0 = ann_ej * STRIDE - OVERLAP
    # Private core top-left in grid coords
    pr_r = r0 + OVERLAP
    pr_c = c0 + OVERLAP
    # Private core rectangle (inside grid)
    prx = ox + pr_c * CELL
    pry = oy + pr_r * CELL
    ax.add_patch(Rectangle((prx, pry), PRIVATE*CELL, PRIVATE*CELL,
                             facecolor='none', edgecolor='#000080',
                             linewidth=1.5, linestyle='-', zorder=6))
    ax.text(prx + PRIVATE*CELL/2, pry + PRIVATE*CELL/2,
            '6×6\nprivate', ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='#000080', zorder=7)

    # Patch outline for expert (0,0)
    patch_x = ox + c0 * CELL
    patch_y = oy + r0 * CELL
    ax.add_patch(Rectangle((patch_x, patch_y), PATCH*CELL, PATCH*CELL,
                             facecolor='none', edgecolor='#222222',
                             linewidth=1.5, linestyle='--', zorder=6))
    ax.text(patch_x + PATCH*CELL/2, patch_y - 0.12,
            '10×10 patch', ha='center', va='top',
            fontsize=7.5, color='#222222', zorder=7)

    # Overlap width annotation
    ax.annotate('', xy=(ox, oy + 2*CELL), xytext=(ox - OVERLAP*CELL, oy + 2*CELL),
                arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.0))
    ax.text(ox - OVERLAP*CELL/2, oy + 2.5*CELL, '2\ncells',
            ha='center', va='bottom', fontsize=7, color='#555555')

    # ── Toroidal wrap arrows (right panel only) ───────────────────────────────
    if not flat_boundary:
        arrowprops = dict(arrowstyle='->', color='#2e7d32', lw=1.5,
                          connectionstyle='arc3,rad=0.35')
        # Left→Right wrap
        ax.annotate('',
                    xy=(ox - OVERLAP*CELL*0.5, oy + GR*CELL*0.5),
                    xytext=(ox + grid_w - OVERLAP*CELL*0.5, oy + GR*CELL*0.5),
                    arrowprops=arrowprops)
        ax.text(ox - OVERLAP*CELL*0.5 - 0.30, oy + GR*CELL*0.5,
                'wraps from right edge', ha='center', va='center', fontsize=7.5,
                color='#2e7d32', fontweight='bold', rotation=90)

        # Top→Bottom wrap
        ax.annotate('',
                    xy=(ox + grid_w/2, oy + grid_h + OVERLAP*CELL*0.6),
                    xytext=(ox + grid_w/2, oy + OVERLAP*CELL*0.6),
                    arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=1.5,
                                    connectionstyle='arc3,rad=0.45'))
        ax.text(ox + grid_w/2, oy + grid_h + OVERLAP*CELL*0.6 + 0.15,
                'wraps from bottom', ha='center', va='bottom', fontsize=7.5,
                color='#2e7d32', fontweight='bold')

    # ── Zero pad label (left panel only) ─────────────────────────────────────
    if flat_boundary:
        ax.text(ox - OVERLAP*CELL/2, oy + GR*CELL/2,
                'Zero\npad', ha='center', va='center', fontsize=8,
                color='#666666', fontweight='bold', rotation=90)
        ax.text(ox + grid_w/2, oy + grid_h + OVERLAP*CELL/2,
                'Zero pad', ha='center', va='center', fontsize=8,
                color='#666666', fontweight='bold')

    # ── Grid dimension labels ─────────────────────────────────────────────────
    ax.text(ox + grid_w/2, oy - 0.38, '24 cols  (768 ÷ 32)',
            ha='center', va='top', fontsize=8, color='#333333')
    ax.text(ox - 0.68, oy + grid_h/2, '32 rows',
            ha='center', va='center', fontsize=8, color='#333333', rotation=90)

    ax.text(ox + grid_w*0.75, oy + grid_h*0.75,
            f'{ER}×{EC} = {ER*EC}\nexperts', ha='center', va='center',
            fontsize=9, color='#333333', fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='#aaaaaa', alpha=0.85, pad=3))

    # ── Legend ────────────────────────────────────────────────────────────────
    ly = oy - 0.90
    items = [
        (darken(ECOLS[1][1], PRIVATE_DARKEN), 0.8, 'Expert private core (6×6)'),
        (ECOLS[1][1], 0.55, 'Shared Expert border (2-cell overlap)'),
        (C_ZERO if flat_boundary else C_WRAP, 0.80,
         'Zero-padding (zeros)' if flat_boundary else 'Toroidal wrap (from opposite edge)'),
    ]
    # Row 1: items 0 and 1
    for i, (fc, al, lbl) in enumerate(items[:2]):
        lx = ox + i * 2.2
        ax.add_patch(Rectangle((lx, ly-0.24), 0.30, 0.22,
                                facecolor=fc, edgecolor='#444444',
                                linewidth=0.6, alpha=al))
        ax.text(lx+0.36, ly-0.13, lbl, va='center', fontsize=8.0)
    # Row 2: item 2
    fc, al, lbl = items[2]
    lx = ox
    ax.add_patch(Rectangle((lx, ly-0.56), 0.30, 0.22,
                            facecolor=fc, edgecolor='#444444',
                            linewidth=0.6, alpha=al))
    ax.text(lx+0.36, ly-0.45, lbl, va='center', fontsize=8.0)

    # View limits
    ax.set_xlim(-1.8, ox + grid_w + 2.0)
    ax.set_ylim(oy - 1.80, oy + grid_h + OVERLAP*CELL + 0.25)

# ── Draw both panels ─────────────────────────────────────────────────────────
draw_panel(axes[0], flat_boundary=True)
draw_panel(axes[1], flat_boundary=False)

axes[0].set_title('Architecture 04 — Flat Grid Overlap\nZero-padding at boundaries',
                   fontsize=13, fontweight='bold', pad=4, color='#111111')
axes[1].set_title('Architecture 05 — RTCC\nToroidal boundary (circular wrap)',
                   fontsize=13, fontweight='bold', pad=4, color='#111111')

fig.text(0.5, 0.975, 'The Single Difference: Boundary Condition',
         ha='center', fontsize=16, fontweight='bold',
         fontfamily='monospace', color='#111111')
fig.text(0.5, 0.945, 'All other hyperparameters identical: 768-dim · 32×24 grid · 10×10 patch · stride 8 · 2-cell overlap · 12 experts · 6×6 private core (36%)',
         ha='center', fontsize=9, color='#555555', style='italic')

plt.subplots_adjust(top=0.84, bottom=0.04, left=0.02, right=0.98, wspace=-0.18)
plt.savefig('figures/rtcc_04_vs_05.pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('figures/rtcc_04_vs_05.png', dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
print("Saved.")
