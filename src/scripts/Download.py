import os
import requests
import shutil
import rasterio
from rasterio.windows import from_bounds, Window
from pyproj import Transformer

# ========== Configuration ==========

# Bounding box in WGS84 (EPSG:4326)
min_lon, min_lat, max_lon, max_lat = -121.67, 39.71, -121.47, 39.91

# Directories
RAW_DIR = "rasters_raw"
CROP_DIR = "cropped_raster"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)

# Remote base URL
BASE_URL = "https://data.pyrecast.org/fuels_and_topography/ca-2021-fuelscape/"
FILE_MAP = {
    "asp":  "asp.tif",
    "cbd":  "cbd.tif",
    "cbh":  "cbh.tif",
    "cc":   "cc.tif",
    "ch":   "ch.tif",
    "dem":  "dem.tif",
    "slp":  "slp.tif",
    "fbfm": "fbfm40.tif"
}

# Tile sizing
TILE_SIZE = 50
OVERLAP   = 25
MAX_TILES = 10000

# ========== Helper ==========

def reproject_bounds(src, bounds4326):
    """Reprojects WGS84 bounds to raster CRS."""
    if src.crs.to_epsg() != 4326:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        min_x, min_y = transformer.transform(bounds4326[0], bounds4326[1])
        max_x, max_y = transformer.transform(bounds4326[2], bounds4326[3])
        return min_x, min_y, max_x, max_y
    return bounds4326

# ========== 1) Download all rasters if not already present ==========

downloaded = {}
for short, fname in FILE_MAP.items():
    out_path = os.path.join(RAW_DIR, fname)
    if os.path.exists(out_path):
        print(f"✔️  Already have {fname}, skipping download.")
    else:
        print(f"⬇️  Downloading {fname} → {out_path}")
        resp = requests.get(BASE_URL + fname, stream=True)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            shutil.copyfileobj(resp.raw, f)
    downloaded[short] = out_path

# ========== 2) Compute windows inside the bbox on 'dem' ==========

ref_path = downloaded["dem"]
with rasterio.open(ref_path) as src:
    # find pixel bounds of our lat/lon box
    bx = reproject_bounds(src, (min_lon, min_lat, max_lon, max_lat))
    window_bbox = from_bounds(*bx, transform=src.transform)
    
    row_start = int(window_bbox.row_off)
    col_start = int(window_bbox.col_off)
    row_stop  = row_start + int(window_bbox.height)
    col_stop  = col_start + int(window_bbox.width)
    
    windows = []
    idx = 0
    for r in range(row_start, row_stop - TILE_SIZE + 1, OVERLAP):
        for c in range(col_start, col_stop - TILE_SIZE + 1, OVERLAP):
            windows.append((idx, r, c))
            idx += 1
            if idx >= MAX_TILES:
                break
        if idx >= MAX_TILES:
            break

print(f"ℹ️  Prepared {len(windows)} windows inside bounding box")

# ========== 3) Crop & save each raster ==========
for short, path in downloaded.items():
    print(f"\n🔪  Cropping {short}.tif → up to {len(windows)} tiles")
    with rasterio.open(path) as src:
        prof = src.profile.copy()
        prof.update({
            "height": TILE_SIZE,
            "width":  TILE_SIZE,
            "transform": None
        })

        for idx, row_off, col_off in windows:
            win = Window(col_off, row_off, TILE_SIZE, TILE_SIZE)

            # read masked: nodata → mask=True
            data = src.read(1, window=win, masked=True)

            # skip any tile that isn’t fully valid
            if data.shape != (TILE_SIZE, TILE_SIZE) or data.mask.any():
                continue

            prof["transform"] = src.window_transform(win)
            out_name = f"cropped_raster_{idx}_{short}.tif"
            os.makedirs(CROP_DIR + f"/{short}", exist_ok=True)
            out_path = os.path.join(CROP_DIR, short, out_name)
            with rasterio.open(out_path, "w", **prof) as dst:
                # .data extracts the underlying numpy array
                dst.write(data.data, 1)

    print(f"✅  Finished {short}")
print("\n🏁 All done! You have up to 10,000 tiles per raster in", CROP_DIR)

