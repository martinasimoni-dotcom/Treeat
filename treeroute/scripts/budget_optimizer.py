import json
import numpy as np
import geopandas as gpd
import osmnx as ox
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import LineString, Point, MultiPoint

DATA = Path(__file__).parent.parent / "backend" / "data"

# ── 1. Load UTCI grid ─────────────────────────────────────────────────────────
grid = np.load(DATA / "utci_leopoldstadt.npy")
# bounds from test_simulation.py result
min_lon, min_lat = 16.375, 48.210
max_lon, max_lat = 16.395708331616788, 48.22839741286381
n_rows, n_cols = grid.shape  # (2048, 1536)

def lonlat_to_pixel(lon, lat):
    col = int(np.clip((lon - min_lon) / (max_lon - min_lon) * (n_cols - 1), 0, n_cols - 1))
    row = int(np.clip((lat - min_lat) / (max_lat - min_lat) * (n_rows - 1), 0, n_rows - 1))
    return row, col

def sample_utci_along_line(coords, n_samples=20):
    """Sample UTCI at n evenly-spaced points along a list of (lon, lat) coords."""
    line = LineString(coords)
    if line.length == 0:
        return np.nan
    distances = np.linspace(0, line.length, n_samples)
    values = []
    for d in distances:
        pt = line.interpolate(d)
        r, c = lonlat_to_pixel(pt.x, pt.y)
        v = grid[r, c]
        if not np.isnan(v):
            values.append(v)
    return float(np.mean(values)) if values else np.nan

# ── 2. Fetch pedestrian street network ────────────────────────────────────────
print("Fetching street network from OSMnx...")
GRAPH_CACHE = DATA / "leopoldstadt_walk.graphml"
if GRAPH_CACHE.exists():
    G = ox.load_graphml(GRAPH_CACHE)
    print("  Loaded from cache")
else:
    G = ox.graph_from_place("Leopoldstadt, Vienna, Austria", network_type="walk")
    ox.save_graphml(G, filepath=GRAPH_CACHE)
    print("  Downloaded and cached")
edges = ox.graph_to_gdfs(G, nodes=False)
edges = edges.to_crs("EPSG:4326")
print(f"  {len(edges)} edges loaded")

# ── 3. Load existing trees ────────────────────────────────────────────────────
print("Loading Baumkataster...")
trees_gdf = gpd.read_file(DATA / "baumkataster_leopoldstadt.geojson")
trees_gdf = trees_gdf.to_crs("EPSG:32633")  # project to metres for buffer
print(f"  {len(trees_gdf)} trees loaded")

edges_m = edges.to_crs("EPSG:32633")  # edges also in metres

# ── 4. Score each edge ────────────────────────────────────────────────────────
print("Scoring edges...")
TREE_BUFFER_M = 15
scores, avg_utcis, tree_counts, lengths_m = [], [], [], []

for idx, row in edges_m.iterrows():
    geom_m = row.geometry
    length_m = geom_m.length

    # count trees within 15 m
    buf = geom_m.buffer(TREE_BUFFER_M)
    nearby = trees_gdf[trees_gdf.geometry.within(buf)]
    n_trees = len(nearby)

    # sample UTCI in geographic coords
    geom_geo = edges.loc[idx].geometry
    coords = list(geom_geo.coords)
    avg_u = sample_utci_along_line(coords)

    if np.isnan(avg_u):
        score = 0.0
    else:
        score = avg_u * length_m * (1 / (1 + n_trees))

    scores.append(score)
    avg_utcis.append(avg_u)
    tree_counts.append(n_trees)
    lengths_m.append(length_m)

edges_m = edges_m.copy()
edges_m["score"]      = scores
edges_m["avg_utci"]   = avg_utcis
edges_m["tree_count"] = tree_counts
edges_m["length_m"]   = lengths_m

# ── 5. Budget allocation ──────────────────────────────────────────────────────
BUDGET       = 50_000
COST_PER_TREE = 2_000
MAX_TREES    = BUDGET // COST_PER_TREE   # 25
SPACING_M    = 8

top_edges = edges_m.nlargest(25, "score")
print(f"\nTop 25 edges selected (budget: €{BUDGET:,} / €{COST_PER_TREE:,} per tree = {MAX_TREES} trees)")

# generate planting points every 8 m along each top edge
planting_points = []
total_trees = 0

for idx, row in top_edges.iterrows():
    if total_trees >= MAX_TREES:
        break
    geom = row.geometry
    length = geom.length
    if length < SPACING_M:
        continue
    n_spots = min(int(length // SPACING_M), MAX_TREES - total_trees)
    for i in range(n_spots):
        pt_m = geom.interpolate(SPACING_M * (i + 0.5))
        planting_points.append({
            "edge_idx": str(idx),
            "geometry": pt_m,
            "avg_utci": row.avg_utci,
            "edge_score": row.score,
        })
        total_trees += 1

planting_gdf = gpd.GeoDataFrame(planting_points, crs="EPSG:32633").to_crs("EPSG:4326")
print(f"Planting locations generated: {len(planting_gdf)}")

# ── 6. Save GeoJSON ───────────────────────────────────────────────────────────
out_geojson = DATA / "planting_locations.geojson"
planting_gdf[["geometry", "avg_utci", "edge_score"]].to_file(out_geojson, driver="GeoJSON")
print(f"Saved -> {out_geojson}")

# ── 7. Visualisation ──────────────────────────────────────────────────────────
print("Rendering visualization...")

fig, ax = plt.subplots(figsize=(11, 14))

# UTCI heatmap background
ax.imshow(
    grid,
    extent=[min_lon, max_lon, min_lat, max_lat],
    origin="lower",
    cmap="RdYlGn_r",
    vmin=np.nanmin(grid),
    vmax=np.nanmax(grid),
    alpha=0.75,
    aspect="auto",
)

# Verification landmarks
ax.plot(16.3930, 48.2187, 'w^', markersize=10, zorder=10)
ax.annotate('Praterstern', (16.3930, 48.2187), color='white', fontsize=7)
ax.plot(16.3785, 48.2245, 'w^', markersize=10, zorder=10)
ax.annotate('Augarten', (16.3785, 48.2245), color='white', fontsize=7)

# all edges (faint grey)
edges_geo = edges.to_crs("EPSG:4326")
for geom in edges_geo.geometry:
    if geom is not None:
        xs, ys = geom.xy
        ax.plot(xs, ys, color="grey", linewidth=0.4, alpha=0.4)

# top candidate edges (blue)
top_geo = top_edges.to_crs("EPSG:4326")
for geom in top_geo.geometry:
    if geom is not None:
        xs, ys = geom.xy
        ax.plot(xs, ys, color="#1a6fbd", linewidth=2.0, alpha=0.9)

# planting spots (green dots)
ax.scatter(
    planting_gdf.geometry.x,
    planting_gdf.geometry.y,
    s=18, color="#2ca02c", zorder=5, label=f"Planting spots ({len(planting_gdf)})",
)

ax.set_xlim(min_lon, max_lon)
ax.set_ylim(min_lat, max_lat)
ax.set_title("Tree Budget Optimizer — Leopoldstadt, Vienna\n"
             f"Budget €{BUDGET:,} · {len(planting_gdf)} trees · top 25 edges highlighted",
             fontsize=13)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend(loc="lower right", fontsize=10)

sm = plt.cm.ScalarMappable(cmap="RdYlGn_r",
                            norm=plt.Normalize(vmin=np.nanmin(grid), vmax=np.nanmax(grid)))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("UTCI (°C)", fontsize=10)

fig.tight_layout()
out_png = DATA / "budget_optimizer.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved -> {out_png}")
