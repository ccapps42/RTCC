"""
RTCC Cortical Column Figure Generator
Generates: figures/rtcc_cortical_column.pdf
Usage: python gen_nested_toruses.py
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

os.makedirs('figures', exist_ok=True)

fig = plt.figure(figsize=(10, 7), facecolor='white')
ax = fig.add_subplot(111, projection='3d', facecolor='white')

R = 2.0
configs = [
    dict(r=1.10, color='#2a6ab0', alpha=0.22, lw=0.7, label='Loop 1', label_color='#2a6ab0'),
    dict(r=0.60, color='#2a9068', alpha=0.50, lw=0.7, label='Loop 2', label_color='#2a9068'),
    dict(r=0.20, color='#b05a10', alpha=0.40, lw=1.1, label='Loop 3', label_color='#5a2000'),
]

NU, NV = 12, 8
u_highlight = 7
v_highlight = 0

u_vals = np.linspace(0, 2*np.pi, NU, endpoint=False)
v_vals = np.linspace(0, 2*np.pi, NV, endpoint=False)
u_fine = np.linspace(0, 2*np.pi, 300)
v_fine = np.linspace(0, 2*np.pi, 100)

def torus_point(u, v, R, r):
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return x, y, z

def get_cell_bounds(u_highlight, v_highlight):
    u0 = u_vals[u_highlight]
    u1_raw = u_vals[(u_highlight + 1) % NU]
    u1 = u1_raw if u1_raw > u0 else u1_raw + 2*np.pi
    v0 = v_vals[v_highlight]
    v1_raw = v_vals[(v_highlight + 1) % NV]
    v1 = v1_raw if v1_raw > v0 else v1_raw + 2*np.pi
    return u0, u1, v0, v1

u0, u1, v0, v1 = get_cell_bounds(u_highlight, v_highlight)

all_corners = []
for cfg in configs:
    r = cfg['r']
    corners = [
        torus_point(u0, v0, R, r),
        torus_point(u0, v1, R, r),
        torus_point(u1, v0, R, r),
        torus_point(u1, v1, R, r),
    ]
    all_corners.append(corners)

for cfg in configs:
    r = cfg['r']
    col = cfg['color']
    al = cfg['alpha']
    lw = cfg['lw']

    for u in u_vals:
        xs, ys, zs = torus_point(u, v_fine, R, r)
        ax.plot(xs, ys, zs, color=col, alpha=al, lw=lw)
    for v in v_vals:
        xs, ys, zs = torus_point(u_fine, v, R, r)
        ax.plot(xs, ys, zs, color=col, alpha=al, lw=lw)

    u_patch = np.linspace(u0, u1, 20)
    v_patch = np.linspace(v0, v1, 12)
    UU, VV = np.meshgrid(u_patch, v_patch)
    XX, YY, ZZ = torus_point(UU, VV, R, r)
    ax.plot_surface(XX, YY, ZZ, color='#f5c400', alpha=0.92, linewidth=0, antialiased=True)

    for u_edge in [u0, u1]:
        xs, ys, zs = torus_point(u_edge, np.linspace(v0, v1, 40), R, r)
        ax.plot(xs, ys, zs, color='#cc3300', lw=2.5, alpha=1.0)
    for v_edge in [v0, v1]:
        xs, ys, zs = torus_point(np.linspace(u0, u1, 40), v_edge, R, r)
        ax.plot(xs, ys, zs, color='#cc3300', lw=2.5, alpha=1.0)

# Column connectors between corners
for ci in range(4):
    for ti in range(len(all_corners) - 1):
        x0c, y0c, z0c = all_corners[ti][ci]
        x1c, y1c, z1c = all_corners[ti+1][ci]
        ax.plot([x0c, x1c], [y0c, y1c], [z0c, z1c],
                color='#555555', lw=1.8, alpha=0.65, linestyle='-', zorder=15)

for corners in all_corners:
    for (x, y, z) in corners:
        ax.scatter([x], [y], [z], color='#555555', s=25, zorder=16, depthshade=False)

# Loop labels — staggered vertically so they don't overlap
z_offsets = [0.75, 0.42, 0.10]  # Loop 1, 2, 3
for cfg, z_off in zip(configs, z_offsets):
    r = cfg['r']
    x, y, z = torus_point(np.pi * 1.1, 0, R, r)
    ax.text(x+0.08, y+0.08, z + z_off, cfg['label'], color=cfg['label_color'],
            fontsize=12, fontweight='bold')

# Expert label
u_mid = (u0 + u1) / 2
v_mid = (v0 + v1) / 2
r_outer = configs[0]['r']
cx, cy, cz = torus_point(u_mid, v_mid, R, r_outer)
angle = u_mid
lx = cx + 0.9 * np.cos(angle)
ly = cy + 0.9 * np.sin(angle)
lz = cz + 0.15
ax.text(lx, ly, lz, r'Expert$_x$',
        color='#884400', fontsize=15, fontweight='bold',
        ha='center', va='bottom', alpha=0.95)
ax.plot([cx, lx], [cy, ly], [cz, lz],
        color='#884400', lw=1.2, alpha=0.6, linestyle='--')

ax.set_axis_off()
ax.set_box_aspect([1, 1, 0.45])
ax.view_init(elev=22, azim=-55)
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)
ax.set_zlim(-1.8, 1.8)

fig.text(0.5, 0.78, 'Recurrent Toroidal Cortical Column',
         ha='center', color='#111111', fontsize=14, fontweight='bold', fontfamily='monospace')
fig.text(0.5, 0.74, r'Same grid cell (Expert$_x$) highlighted across all three recurrent loops',
         ha='center', color='#444444', fontsize=9)

plt.tight_layout()
plt.savefig('figures/rtcc_cortical_column.pdf',
            bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('figures/rtcc_cortical_column.png',
            dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
print("Saved: figures/rtcc_cortical_column.pdf and .png")
