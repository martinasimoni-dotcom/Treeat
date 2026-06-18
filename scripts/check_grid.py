import numpy as np
import osmnx as ox
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "backend" / "data"
NPY = DATA_DIR / "utci_leopoldstadt.npy"
PNG = DATA_DIR / "utci_leopoldstadt.png"

min_lon, min_lat = 16.375, 48.210
max_lon, max_lat = 16.395708331616788, 48.22839741286381

grid = np.load(NPY)
n_rows, n_cols = grid.shape

nan_cells   = int(np.isnan(grid).sum())
valid_cells = int((~np.isnan(grid)).sum())
total_cells = grid.size
utci_min  = float(np.nanmin(grid))
utci_max  = float(np.nanmax(grid))
utci_mean = float(np.nanmean(grid))
pct_heat  = float(np.nansum(grid > 32) / valid_cells * 100)

print(f"Grid shape:    {grid.shape}")
print(f"NaN cells:     {nan_cells:,}  ({nan_cells/total_cells*100:.1f}%)")
print(f"Valid cells:   {valid_cells:,}  ({valid_cells/total_cells*100:.1f}%)")
print(f"UTCI min:      {utci_min:.2f} °C")
print(f"UTCI max:      {utci_max:.2f} °C")
print(f"UTCI mean:     {utci_mean:.2f} °C")
print(f"Above 32 °C:   {pct_heat:.1f}% of valid cells")

# Load street network for overlay
print("Loading street network for overlay...")
GRAPH_CACHE = DATA_DIR / "leopoldstadt_walk.graphml"
G = ox.load_graphml(GRAPH_CACHE)
edges = ox.graph_to_gdfs(G, nodes=False).to_crs("EPSG:4326")

fig, ax = plt.subplots(figsize=(11, 14))

ax.imshow(
    grid,
    extent=[min_lon, max_lon, min_lat, max_lat],
    origin="lower",
    cmap="RdYlGn_r",
    vmin=utci_min,
    vmax=utci_max,
    alpha=0.75,
    aspect="auto",
)

# Street overlay — same approach as budget_optimizer.py
for geom in edges.geometry:
    if geom is not None:
        xs, ys = geom.xy
        ax.plot(xs, ys, color="grey", linewidth=0.4, alpha=0.4)

# Verification landmarks
ax.plot(16.3930, 48.2187, 'w^', markersize=10, zorder=10)
ax.annotate('Praterstern', (16.3930, 48.2187), color='white', fontsize=7)
ax.plot(16.3785, 48.2245, 'w^', markersize=10, zorder=10)
ax.annotate('Augarten', (16.3785, 48.2245), color='white', fontsize=7)

cbar = fig.colorbar(
    plt.cm.ScalarMappable(cmap="RdYlGn_r",
                          norm=plt.Normalize(vmin=utci_min, vmax=utci_max)),
    ax=ax, fraction=0.025, pad=0.02,
)
cbar.set_label("UTCI (°C)", fontsize=11)
ax.set_title("UTCI July — Leopoldstadt, Vienna", fontsize=14, pad=12)
ax.set_xlim(min_lon, max_lon)
ax.set_ylim(min_lat, max_lat)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
fig.tight_layout()
fig.savefig(PNG, dpi=150, bbox_inches="tight")
print(f"\nSaved heatmap -> {PNG}")
