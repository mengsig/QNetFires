import os
import requests
import shutil
import rasterio
from rasterio.windows import Window
from pyproj import Transformer

# ========== Configuration ==========

# Where to download raw .tif files and write cropped outputs
RASters_DIR = "rasters"
os.makedirs(RASters_DIR, exist_ok=True)

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

# ========== Download all rasters ==========

downloaded = {}
for short, fname in FILE_MAP.items():
    url = BASE_URL + fname
    dest = os.path.join(RASters_DIR, fname)
    print(f"⬇️  Downloading {fname} → {dest}")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(dest, "wb") as out:
        shutil.copyfileobj(resp.raw, out)
    downloaded[short] = dest

# ========== Compute windows on reference raster (dem) ==========

ref_path = downloaded["dem"]
with rasterio.open(ref_path) as ref:
    width  = ref.width
    height = ref.height

    windows = []
    idx = 0
    for row_off in range(0, height - TILE_SIZE + 1, OVERLAP):
        for col_off in range(0, width - TILE_SIZE + 1, OVERLAP):
            windows.append((idx, row_off, col_off))
            idx += 1

print(f"ℹ️  Generated {len(windows)} full 50×50 windows")

# ========== Crop & write every raster at each window ==========

for short, path in downloaded.items():
    print(f"\n🔪  Cropping {short}.tif into {len(windows)} tiles…")
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        profile.update({
            "height": TILE_SIZE,
            "width":  TILE_SIZE,
            "transform": None  # will set per-window below
        })
        for idx, row_off, col_off in windows:
            win = Window(col_off, row_off, TILE_SIZE, TILE_SIZE)
            data = src.read(1, window=win)
            # update transform for this window
            profile["transform"] = src.window_transform(win)
            out_name = f"cropped_raster_{idx}_{short}.tif"
            out_path = os.path.join(RASters_DIR, out_name)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data, 1)

    print(f"✅  Finished {short}, files → cropped_raster_*_{short}.tif")

print("\n🏁 All done!")

