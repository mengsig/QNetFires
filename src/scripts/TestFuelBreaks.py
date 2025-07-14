import numpy as np
from FireEnv import FireEnv
import os
import sys
script_dir   = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.insert(0, project_root)
from src.utils.loadingUtils import load_all_rasters

# load or generate your rasters as NumPy arrays:
rasterDict = load_all_rasters("cropped_raster", 50)
slope, aspect, dem, cc, cbd, cbh, ch, fuel_model = rasterDict.values()
env = FireEnv(slope, aspect, dem, cc, cbd, cbh, ch, fuel_model)
reset_result = env.reset()
obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

# Sample random fuel‐break mask:
action = env.action_space.sample()
obs, reward, done, info = env.step(action)

print("Burned acres:", -reward)
env.render()         # pops up the final map

