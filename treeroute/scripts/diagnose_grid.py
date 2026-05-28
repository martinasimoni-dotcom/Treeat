import numpy as np

grid = np.load(r'c:\Users\marts\Documents\10_IAAC\3_third term\Programming for AI\Treeat\treeroute\backend\data\utci_leopoldstadt.npy')
min_lon, min_lat = 16.375, 48.210
max_lon, max_lat = 16.395708331616788, 48.22839741286381
n_rows, n_cols = grid.shape

print(f"Shape: {grid.shape}  (rows=height/lat, cols=width/lon)")
print(f"Cell size: {(max_lon-min_lon)/n_cols*111320*np.cos(np.radians(48.21)):.2f} m/col   {(max_lat-min_lat)/n_rows*111320:.2f} m/row")
print()

# NaN by quadrant
def quad_valid(r0, r1, c0, c1):
    q = grid[r0:r1, c0:c1]
    return (~np.isnan(q)).sum(), q.size

hr, hc = n_rows // 2, n_cols // 2
nw = quad_valid(0,  hr,     0,  hc)
ne = quad_valid(0,  hr,    hc,  n_cols)
sw = quad_valid(hr, n_rows, 0,  hc)
se = quad_valid(hr, n_rows, hc, n_cols)
print("Quadrant valid cells (row 0 = TOP of array):")
print(f"  top-left  (NW if row0=north): {nw[0]:>7,} / {nw[1]:,}  ({100*nw[0]/nw[1]:.0f}%)")
print(f"  top-right (NE if row0=north): {ne[0]:>7,} / {ne[1]:,}  ({100*ne[0]/ne[1]:.0f}%)")
print(f"  bot-left  (SW if row0=north): {sw[0]:>7,} / {sw[1]:,}  ({100*sw[0]/sw[1]:.0f}%)")
print(f"  bot-right (SE if row0=north): {se[0]:>7,} / {se[1]:,}  ({100*se[0]/se[1]:.0f}%)")
print()

# Edge rows/cols
print(f"Row 0   valid pixels: {int((~np.isnan(grid[0,:])).sum())} / {n_cols}")
print(f"Row -1  valid pixels: {int((~np.isnan(grid[-1,:])).sum())} / {n_cols}")
print(f"Col 0   valid pixels: {int((~np.isnan(grid[:,0])).sum())} / {n_rows}")
print(f"Col -1  valid pixels: {int((~np.isnan(grid[:,-1])).sum())} / {n_rows}")
print()

# Landmark lookup under both orientation assumptions
def px_north_up(lon, lat):
    c = int(np.clip((lon - min_lon) / (max_lon - min_lon) * (n_cols - 1), 0, n_cols - 1))
    r = int(np.clip((max_lat - lat) / (max_lat - min_lat) * (n_rows - 1), 0, n_rows - 1))
    return r, c

def px_south_up(lon, lat):
    c = int(np.clip((lon - min_lon) / (max_lon - min_lon) * (n_cols - 1), 0, n_cols - 1))
    r = int(np.clip((lat - min_lat) / (max_lat - min_lat) * (n_rows - 1), 0, n_rows - 1))
    return r, c

landmarks = [
    ("Praterstern",      16.3930, 48.2187),
    ("Augarten edge",    16.3785, 48.2245),
    ("Karmeliterplatz",  16.3812, 48.2183),
    ("Polygon SW corner",16.3750, 48.2100),
    ("Polygon NE corner",16.3950, 48.2250),
]
print("Landmark UTCI under each orientation:")
print(f"  {'Name':<22} {'row0=north':^22} {'row0=south':^22}")
print("  " + "-"*66)
for name, lon, lat in landmarks:
    rn, cn = px_north_up(lon, lat)
    rs, cs = px_south_up(lon, lat)
    vn = grid[rn, cn]
    vs = grid[rs, cs]
    sn = "NaN" if np.isnan(vn) else f"{vn:.2f} C"
    ss = "NaN" if np.isnan(vs) else f"{vs:.2f} C"
    print(f"  {name:<22} row={rn:4d} col={cn:4d} {sn:>8}   row={rs:4d} col={cs:4d} {ss:>8}")
print()

# Where is the data centroid?
valid_rows = np.where((~np.isnan(grid)).any(axis=1))[0]
valid_cols = np.where((~np.isnan(grid)).any(axis=0))[0]
rc = (valid_rows[0] + valid_rows[-1]) // 2
cc = (valid_cols[0] + valid_cols[-1]) // 2
print(f"Valid data spans rows {valid_rows[0]}–{valid_rows[-1]}, cols {valid_cols[0]}–{valid_cols[-1]}")
print(f"Data centroid pixel: row={rc}, col={cc}")
lat_n = max_lat - rc / (n_rows - 1) * (max_lat - min_lat)
lat_s = min_lat + rc / (n_rows - 1) * (max_lat - min_lat)
lon_c = min_lon + cc / (n_cols - 1) * (max_lon - min_lon)
print(f"  -> if row0=north: lat={lat_n:.4f}, lon={lon_c:.4f}  (expected ~48.218, ~16.385)")
print(f"  -> if row0=south: lat={lat_s:.4f}, lon={lon_c:.4f}  (expected ~48.218, ~16.385)")
