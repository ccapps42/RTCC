"""
RTCC Expert Overlap Figure
Curved outer surface of one torus section showing 9x9 expert patch,
7x7 private core (60%), and 1-cell shared border with neighbors.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

os.makedirs('figures', exist_ok=True)

fig = plt.figure(figsize=(12, 8), facecolor='white')
ax = fig.add_subplot(111, projection='3d', facecolor='white')

R = 3.0
r = 3.5

EXPERT = 9
NBOR   = 2
TOTAL  = EXPERT + 2 * NBOR  # 13

v_half = 0.75
u_half = 0.40

v_edges = np.linspace(-v_half, v_half, TOTAL + 1)
u_edges = np.linspace(-u_half, u_half, TOTAL + 1)

def tp(u, v):
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return x, y, z

def draw_cell(ui, vi, color, alpha, n=10):
    ug = np.linspace(u_edges[ui], u_edges[ui+1], n)
    vg = np.linspace(v_edges[vi], v_edges[vi+1], n)
    UU, VV = np.meshgrid(ug, vg)
    X, Y, Z = tp(UU, VV)
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0, antialiased=True)

ce_u0, ce_u1 = NBOR, NBOR + EXPERT
ce_v0, ce_v1 = NBOR, NBOR + EXPERT
pc_u0, pc_u1 = ce_u0+1, ce_u1-1
pc_v0, pc_v1 = ce_v0+1, ce_v1-1

core_color   = '#3a7ecf'
shared_color = '#f0c000'
nbor_L = '#4caf50'
nbor_R = '#e05c4e'
nbor_T = '#9c5cbf'
nbor_B = '#e8802a'
corner_color = '#aaaaaa'

for vi in range(TOTAL):
    for ui in range(TOTAL):
        in_center  = (ce_u0 <= ui < ce_u1) and (ce_v0 <= vi < ce_v1)
        in_private = (pc_u0 <= ui < pc_u1) and (pc_v0 <= vi < pc_v1)
        on_border  = in_center and not in_private
        left   = ui < ce_u0
        right  = ui >= ce_u1
        below  = vi < ce_v0
        above  = vi >= ce_v1

        # Corner border cells (shared by 3 experts) get darker yellow
        is_border_corner = on_border and (ui in (ce_u0, ce_u1-1)) and (vi in (ce_v0, ce_v1-1))

        if in_private:
            draw_cell(ui, vi, core_color, 0.65)
        elif is_border_corner:
            draw_cell(ui, vi, '#b07800', 0.95)   # dark amber — 3-expert overlap
        elif on_border:
            draw_cell(ui, vi, shared_color, 0.85)
        elif (left or right) and (above or below):
            draw_cell(ui, vi, corner_color, 0.20)
        elif left:
            draw_cell(ui, vi, nbor_L, 0.30)
        elif right:
            draw_cell(ui, vi, nbor_R, 0.30)
        elif above:
            draw_cell(ui, vi, nbor_T, 0.30)
        elif below:
            draw_cell(ui, vi, nbor_B, 0.30)

# Bold lines: outer edges of center expert (blue) + inner edges of neighbors (orange = overlap boundary)
outer_u = {ce_u0, ce_u1}          # center expert outer edges
inner_u = {pc_u0, pc_u1}          # inner boundary of overlap strip = neighbor outer edges
outer_v = {ce_v0, ce_v1}
inner_v = {pc_v0, pc_v1}

for ui in range(TOTAL + 1):
    vg = np.linspace(-v_half, v_half, 80)
    X, Y, Z = tp(u_edges[ui], vg)
    if ui in outer_u:
        col, lw, al = '#1a4a80', 1.6, 1.0   # blue — center expert boundary
    elif ui in inner_u:
        col, lw, al = '#c07000', 1.3, 0.9   # amber — neighbor outer / overlap inner boundary
    else:
        col, lw, al = '#333333', 0.45, 0.45
    ax.plot(X, Y, Z, color=col, lw=lw, alpha=al)

for vi in range(TOTAL + 1):
    ug = np.linspace(-u_half, u_half, 100)
    X, Y, Z = tp(ug, v_edges[vi])
    if vi in outer_v:
        col, lw, al = '#1a4a80', 1.6, 1.0
    elif vi in inner_v:
        col, lw, al = '#c07000', 1.3, 0.9
    else:
        col, lw, al = '#333333', 0.45, 0.45
    ax.plot(X, Y, Z, color=col, lw=lw, alpha=al)

# Shared border label
u_sh = (u_edges[ce_u0] + u_edges[ce_u0+1]) / 2
v_sh = 0.0
xs, ys, zs = tp(u_sh, v_sh)
xl, yl, zl = xs - 0.8, ys - 1.0, zs + 1.2
ax.text(xl, yl, zl, 'Shared\nBorder\n(1 cell)', color='#8a6000',
        fontsize=9, fontweight='bold', ha='center',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1.5))
ax.plot([xl, xs+0.05], [yl+0.3, ys], [zl-0.3, zs+0.05],
        color='#8a6000', lw=1.0, alpha=0.7, linestyle='--')

# Neighbor labels
for (u_idx, v_idx, txt, col, z_off) in [
    (0,  6,  'Neighbor\n(left)',   nbor_L,  0.08),
    (12, 6,  'Neighbor\n(right)',  nbor_R,  0.08),
    (6,  12, 'Neighbor\n(top)',    nbor_T,  0.08),
    (6,  0,  'Neighbor\n(bottom)', nbor_B, -0.60),
]:
    uc = (u_edges[u_idx] + u_edges[min(u_idx+1, TOTAL)]) / 2
    vc = (v_edges[v_idx] + v_edges[min(v_idx+1, TOTAL)]) / 2
    xn, yn, zn = tp(uc, vc)
    ax.text(xn, yn, zn + z_off, txt, color=col,
            fontsize=8, fontweight='bold', ha='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.65, pad=1))

ax.set_axis_off()
ax.set_position([0.05, 0.05, 0.85, 0.90])  # shift plot left
ax.set_box_aspect([1.0, 1.0, 0.55])
ax.view_init(elev=32, azim=-20)
ax.set_xlim(-1.0, 6.8)
ax.set_ylim(-3.0, 3.0)
ax.set_zlim(-2.5, 2.5)

fig.text(0.5, 0.67, 'RTCC Expert Patches on Toroidal Surface',
         ha='center', color='#111111', fontsize=14, fontweight='bold', fontfamily='monospace')
fig.text(0.5, 0.63, '9x9 patch  |  7x7 private core (60%)  |  1-cell shared border with each neighbor',
         ha='center', color='#444444', fontsize=9)

# Expert_x label — drawn last so it's on top of everything
fig.text(0.54, 0.37, r'Expert$_x$',
         ha='center', va='center', color='black', fontsize=15, fontweight='bold',
         bbox=dict(facecolor='white', edgecolor='#1a4a80', linewidth=1.5,
                   alpha=1.0, pad=5, boxstyle='round,pad=0.4'))

plt.savefig('figures/rtcc_expert_overlap.pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('figures/rtcc_expert_overlap.png', dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
print("Saved.")
