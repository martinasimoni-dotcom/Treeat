import numpy as np
import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from shapely.geometry import LineString

DATA = Path(__file__).parent.parent / "backend" / "data"

# ── 1. Load UTCI grid and amplify contrast ────────────────────────────────────
grid_raw = np.load(DATA / "utci_leopoldstadt.npy")
min_lon, min_lat = 16.375, 48.210
max_lon, max_lat = 16.395708331616788, 48.22839741286381
n_rows, n_cols = grid_raw.shape

grid = np.where(
    np.isnan(grid_raw), np.nan,
    22 + (grid_raw - np.nanmin(grid_raw)) /
         (np.nanmax(grid_raw) - np.nanmin(grid_raw)) * 15
)

def lonlat_to_pixel(lon, lat):
    col = int(np.clip((lon - min_lon) / (max_lon - min_lon) * (n_cols - 1), 0, n_cols - 1))
    row = int(np.clip((lat - min_lat) / (max_lat - min_lat) * (n_rows - 1), 0, n_rows - 1))
    return row, col

def sample_utci_along_coords(coords, n_samples=20):
    line = LineString(coords)
    if line.length == 0:
        return float(np.nanmean(grid[~np.isnan(grid)]))
    distances = np.linspace(0, line.length, n_samples)
    values = []
    for d in distances:
        pt = line.interpolate(d)
        r, c = lonlat_to_pixel(pt.x, pt.y)
        v = grid[r, c]
        if not np.isnan(v):
            values.append(v)
    return float(np.mean(values)) if values else float(np.nanmean(grid))

# ── 2. Load pedestrian graph (WGS84) ─────────────────────────────────────────
print("Loading street network...")
GRAPH_CACHE = DATA / "leopoldstadt_walk.graphml"
if GRAPH_CACHE.exists():
    G = ox.load_graphml(GRAPH_CACHE)
    print("  Loaded from cache")
else:
    print("  Downloading via place name...")
    G = ox.graph_from_place("Leopoldstadt, Vienna, Austria", network_type="walk")
    ox.save_graphml(G, filepath=GRAPH_CACHE)
print(f"  {G.number_of_edges()} edges, {G.number_of_nodes()} nodes")

# ── 3. Demo points ────────────────────────────────────────────────────────────
ORIGIN      = (16.3812, 48.2183)   # Karmeliterplatz
DESTINATION = (16.3930, 48.2187)   # Praterstern

# Snap directly on the WGS84 graph — OSMnx handles this correctly
orig_node, dist_orig = ox.nearest_nodes(G, ORIGIN[0],      ORIGIN[1],      return_dist=True)
dest_node, dist_dest = ox.nearest_nodes(G, DESTINATION[0], DESTINATION[1], return_dist=True)
print(f"  Origin snapped {dist_orig:.0f} m from input point")
print(f"  Destination snapped {dist_dest:.0f} m from input point")
if dist_orig > 100 or dist_dest > 100:
    print("  WARNING: snap distance >100 m — point may be outside the network")

# ── 4. Sample UTCI for every edge ────────────────────────────────────────────
print("Sampling UTCI on edges...")
for u, v, key, data in G.edges(keys=True, data=True):
    if "geometry" in data:
        coords = list(data["geometry"].coords)
    else:
        coords = [(G.nodes[u]["x"], G.nodes[u]["y"]),
                  (G.nodes[v]["x"], G.nodes[v]["y"])]
    data["utci"] = sample_utci_along_coords(coords)

# ── 5. Find routes ────────────────────────────────────────────────────────────
print("Computing routes...")
path_short = nx.shortest_path(G, orig_node, dest_node, weight="length")
path_cool  = nx.shortest_path(G, orig_node, dest_node, weight="utci")
print(f"  Shortest path: {len(path_short)-1} edges")
print(f"  Coolest path:  {len(path_cool)-1} edges")

# ── 6. Extract actual edge geometries for plotting (same as budget_optimizer) ─
def route_to_edge_geoms(G, path):
    """Return list of LineString geometries from edge data — follows real streets."""
    geoms = []
    for u, v in zip(path[:-1], path[1:]):
        edge = min(G[u][v].values(), key=lambda d: d.get("length", 0))
        if "geometry" in edge:
            geoms.append(edge["geometry"])
        else:
            from shapely.geometry import LineString
            geoms.append(LineString([
                (G.nodes[u]["x"], G.nodes[u]["y"]),
                (G.nodes[v]["x"], G.nodes[v]["y"]),
            ]))
    return geoms

shortest_geoms = route_to_edge_geoms(G, path_short)
coolest_geoms  = route_to_edge_geoms(G, path_cool)

# ── 7. Compute metrics ────────────────────────────────────────────────────────
def route_metrics(G, path):
    total_dist, total_thermal = 0.0, 0.0
    utci_vals = []
    for u, v in zip(path[:-1], path[1:]):
        edge = min(G[u][v].values(), key=lambda d: d.get("length", 0))
        length = edge.get("length", 0)
        utci   = edge.get("utci", np.nan)
        total_dist    += length
        total_thermal += utci * length
        utci_vals.append(utci)
    avg_utci = float(np.nanmean(utci_vals)) if utci_vals else np.nan
    return total_dist, avg_utci, total_thermal

dist_s, utci_s, thermal_s = route_metrics(G, path_short)
dist_c, utci_c, thermal_c = route_metrics(G, path_cool)

# ── 8. Print comparison table ─────────────────────────────────────────────────
print()
print(f"{'Metric':<28} {'Shortest':>12} {'Coolest':>12}")
print("-" * 54)
print(f"{'Distance (m)':<28} {dist_s:>12.1f} {dist_c:>12.1f}")
print(f"{'Avg UTCI (°C)':<28} {utci_s:>12.2f} {utci_c:>12.2f}")
print(f"{'Thermal load (°C·m)':<28} {thermal_s:>12.1f} {thermal_c:>12.1f}")
print()
print(f"Coolest route is {utci_s - utci_c:+.2f} °C cooler on average")
print(f"Coolest route is {dist_c - dist_s:+.1f} m longer")

# ── 9. Visualisation ──────────────────────────────────────────────────────────
print("\nRendering visualization...")

fig, ax = plt.subplots(figsize=(11, 14))

# UTCI heatmap — explicit WGS84 extent
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

# Street network — same approach as budget_optimizer.py (no clipping)
_, edges_gdf = ox.graph_to_gdfs(G)
edges_gdf = edges_gdf.to_crs("EPSG:4326")
for geom in edges_gdf.geometry:
    if geom is not None:
        xs, ys = geom.xy
        ax.plot(xs, ys, color="grey", linewidth=0.4, alpha=0.4)

# Routes — drawn from real edge geometries, same as budget_optimizer top_edges
for geom in shortest_geoms:
    xs, ys = geom.xy
    ax.plot(xs, ys, color="#e31a1c", linewidth=2.5, zorder=5,
            solid_capstyle="round", solid_joinstyle="round")
for geom in coolest_geoms:
    xs, ys = geom.xy
    ax.plot(xs, ys, color="#1f78b4", linewidth=2.5, zorder=6,
            solid_capstyle="round", solid_joinstyle="round")

# Markers
ax.scatter(*ORIGIN,      s=140, color="#2ca02c", zorder=9, edgecolors="white", linewidths=1.5)
ax.scatter(*DESTINATION, s=140, color="#d62728", zorder=9, edgecolors="white", linewidths=1.5)
ax.annotate("Origin\n(Karmeliterplatz)", xy=ORIGIN,
            xytext=(6, 4), textcoords="offset points", fontsize=8,
            color="white", fontweight="bold", zorder=10)
ax.annotate("Destination\n(Praterstern)", xy=DESTINATION,
            xytext=(6, -18), textcoords="offset points", fontsize=8,
            color="white", fontweight="bold", zorder=10)

# Legend
legend_handles = [
    mpatches.Patch(color="#e31a1c", label=f"Fastest route — {dist_s:.0f} m · feels like {utci_s:.1f}°C"),
    mpatches.Patch(color="#1f78b4", label=f"Coolest route — {dist_c:.0f} m · feels like {utci_c:.1f}°C"),
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=9,
          framealpha=0.85, edgecolor="grey")

# Trade-off annotation
delta_m   = dist_c - dist_s
delta_deg = utci_s - utci_c
mid_lon = (ORIGIN[0] + DESTINATION[0]) / 2
mid_lat = (ORIGIN[1] + DESTINATION[1]) / 2
ax.annotate(
    f"+{delta_m:.0f} m longer but {delta_deg:.2f}°C cooler",
    xy=(mid_lon, mid_lat),
    xytext=(10, 18), textcoords="offset points",
    fontsize=9, color="white", fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", fc="#1f78b4", alpha=0.75, ec="none"),
    zorder=11,
)

sm = plt.cm.ScalarMappable(cmap="RdYlGn_r",
                            norm=plt.Normalize(vmin=np.nanmin(grid), vmax=np.nanmax(grid)))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("UTCI (°C)", fontsize=10)

# Verification landmarks
ax.plot(16.3930, 48.2187, 'w^', markersize=10, zorder=10)
ax.annotate('Praterstern', (16.3930, 48.2187), color='white', fontsize=7)
ax.plot(16.3785, 48.2245, 'w^', markersize=10, zorder=10)
ax.annotate('Augarten', (16.3785, 48.2245), color='white', fontsize=7)

ax.set_xlim(min_lon, max_lon)
ax.set_ylim(min_lat, max_lat)
ax.set_title("Cool Route — Leopoldstadt, Vienna\nFastest (red) vs Coolest (blue)", fontsize=13)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
fig.tight_layout()
fig.savefig(DATA / "cool_route.png", dpi=150, bbox_inches="tight")
print(f"Saved -> {DATA / 'cool_route.png'}")

# ── 10. Save coolest route GeoJSON ────────────────────────────────────────────
cool_gdf = gpd.GeoDataFrame(
    [{"route": "coolest", "distance_m": dist_c, "avg_utci": utci_c, "thermal_load": thermal_c}],
    geometry=[LineString([pt for g in coolest_geoms for pt in g.coords])],
    crs="EPSG:4326",
)
cool_gdf.to_file(DATA / "cool_route.geojson", driver="GeoJSON")
print(f"Saved -> {DATA / 'cool_route.geojson'}")
