#global imports
import os
import sys
import numpy as np
#pyretechnics import
from pyretechnics.space_time_cube import SpaceTimeCube
import pyretechnics.burn_cells as bc

#src/utils/ imports
script_dir   = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.insert(0, project_root)
from src.utils.loadingUtils import (
    load_raster,
    normalize,
        )
from src.utils.plottingUtils import (
    save_matrix_as_heatmap,
    )
from src.utils.parsingUtils import (
    parse_args,
    )
from src.utils.networkUtils import (
    build_edgelist_from_spread_rates,
    )

# -------------Tunable parameters------------- #
xlen,ylen, savename, _, _ = parse_args()
time_steps = 24
# ----------end of tunable parameters----------#

savedir = f"results/{savename}"
print(f"[CREATE-ADJACENCY]: Saving to {savename}...")
os.makedirs(savedir, exist_ok = True)

#directory containing the already-cropped rasters
raster_dir = "cropped_rasters"

#load rasters 
xoffset = 100
yoffset = 100
xsubset = (xoffset,int(xoffset + xlen))
ysubset = (yoffset,int(yoffset + ylen))

slope = load_raster("slp", xsubset, ysubset)
aspect = load_raster("asp", xsubset, ysubset)
dem = load_raster("dem", xsubset, ysubset)
cc = load_raster("cc", xsubset, ysubset) 
cbd = load_raster("cbd", xsubset, ysubset) 
cbh = load_raster("cbh", xsubset, ysubset)
ch = load_raster("ch", xsubset, ysubset)
fuel_model = load_raster("fbfm", xsubset, ysubset)

#convert to cubes
slope_cube = normalize(slope, time_steps, datatype = "slp")
aspect_cube = normalize(aspect, time_steps, datatype = "asp")
dem_cube   = normalize(dem, time_steps, datatype = "dem")
cc_cube    = normalize(cc, time_steps, datatype = "cc")
cbd_cube   = normalize(cbd, time_steps, datatype = "cbd")
cbh_cube   = normalize(cbh, time_steps, datatype = "cbh")
ch_cube    = normalize(ch, time_steps, datatype = "ch")
fuel_model_cube    = normalize(fuel_model, time_steps, datatype = "fbfm")

# Define cube shape (e.g. 24 hours)
rows, cols = slope.shape
cube_shape = (time_steps, rows, cols)

#build space-time cubes
space_time_cubes = {
    "slope"                        : SpaceTimeCube(cube_shape, slope_cube),
    "aspect"                       : SpaceTimeCube(cube_shape, aspect_cube),
    "fuel_model"                   : SpaceTimeCube(cube_shape, fuel_model),
    "canopy_cover"                 : SpaceTimeCube(cube_shape, cc_cube),
    "canopy_height"                : SpaceTimeCube(cube_shape, ch_cube),
    "canopy_base_height"           : SpaceTimeCube(cube_shape, cbh_cube),
    "canopy_bulk_density"          : SpaceTimeCube(cube_shape, cbd_cube),
    "wind_speed_10m"               : SpaceTimeCube(cube_shape, 0),
    "upwind_direction"             : SpaceTimeCube(cube_shape, 0),
    "fuel_moisture_dead_1hr"       : SpaceTimeCube(cube_shape, 0.10),
    "fuel_moisture_dead_10hr"      : SpaceTimeCube(cube_shape, 0.25),
    "fuel_moisture_dead_100hr"     : SpaceTimeCube(cube_shape, 0.50),
    "fuel_moisture_live_herbaceous": SpaceTimeCube(cube_shape, 0.90),
    "fuel_moisture_live_woody"     : SpaceTimeCube(cube_shape, 0.60),
    "foliar_moisture"              : SpaceTimeCube(cube_shape, 0.90),
    "fuel_spread_adjustment"       : SpaceTimeCube(cube_shape, 1.0),
    "weather_spread_adjustment"    : SpaceTimeCube(cube_shape, 1.0),
}
#============================================================================================
# Specify a space-time region (t, (y_min, y_max), (x_min, x_max))
# within the extent of the SpaceTimeCube dimensions
#============================================================================================

t = cube_shape[0]
y = cube_shape[1]
x = cube_shape[2]
y_range = (0,y)
x_range = (0,x)
directions = 4 
azimuth_step = 360/directions
spread_rate_mean = np.zeros((directions,y,x))
spread_azimuth = 0 # degrees clockwise from North on the horizontal plane
#orig_space_time_cubes = space_time_cubes.copy()
for i in range(directions):
    num_simulations = 0
    for step in range(t):
        if step % directions != 0:
            continue
        print(f"[CREATE-ADJACENCY]: direction {i}/{directions}, step: {step}/{t}")
        num_simulations += 1
#============================================================================================
# Calculate combined fire behavior in the direction of the azimuth (with wind limit)
#============================================================================================

        combined_behavior_limited = bc.burn_all_cells_toward_azimuth(space_time_cubes,
                                                                     spread_azimuth,
                                                                     step,
                                                                     y_range,
                                                                     x_range,
                                                                     surface_lw_ratio_model="behave")

#============================================================================================
# Calculate combined fire behavior in the direction of the azimuth (without wind limit)
#============================================================================================

        combined_behavior_unlimited = bc.burn_all_cells_toward_azimuth(space_time_cubes,
                                                                       spread_azimuth,
                                                                       step,
                                                                       y_range,
                                                                       x_range,
                                                                       use_wind_limit=False,
                                                                       surface_lw_ratio_model="behave")

#============================================================================================
# TODO: do we use unlimited or limited?
#============================================================================================
        spread_rate_mean[i] += combined_behavior_unlimited["fireline_intensity"]

#============================================================================================
# Update spread azimuth angle
#============================================================================================
    spread_azimuth += azimuth_step

    #normalize spread_rate_mean
    spread_rate_mean[i,::,::] /= num_simulations

edgelist = build_edgelist_from_spread_rates(spread_rate_mean, x, y)
edgelist_array = np.array(edgelist, dtype=np.float32)
np.savetxt(f"{savedir}/spread_edge_list.txt", edgelist_array)

#============================================================================================
# Display combined fire behavior in the direction of the azimuth (with wind limit)
#============================================================================================

spread_rate_mean = spread_rate_mean + 1
vmin = spread_rate_mean.min()
vmax = spread_rate_mean.max()

heatmap_configs = [
    {
        "matrix"  : spread_rate_mean[0,::,::],
        "colors"  : "hot",
        "units"   : "m/min",
        "title"   : "Fireline Intensity North",
        "filename": f"{savedir}/adjacency_north.png",
        "vmin"    : vmin,
        "vmax"    : vmax,
        "norm"    : True,
    },
    {
        "matrix"  : spread_rate_mean[1,::,::],
        "colors"  : "hot",
        "units"   : "m/min",
        "title"   : "Fireline Intensity East",
        "filename": f"{savedir}/adjacency_east.png",
        "vmin"    : vmin,
        "vmax"    : vmax,
        "norm"    : True,
    },
    {
        "matrix"  : spread_rate_mean[2,::,::],
        "colors"  : "hot",
        "units"   : "m/min",
        "title"   : "Fireline Intensity South",
        "filename": f"{savedir}/adjacency_south.png",
        "vmin"    : vmin,
        "vmax"    : vmax,
        "norm"    : True,
    },
    {
        "matrix"  : spread_rate_mean[3,::,::],
        "colors"  : "hot",
        "units"   : "m/min",
        "title"   : "Fireline Intensity West",
        "filename": f"{savedir}/adjacency_west.png",
        "vmin"    : vmin,
        "vmax"    : vmax,
        "norm"    : True,
    },
]


#saving north/west/south/east heatmaps
for heatmap_config in heatmap_configs:
    save_matrix_as_heatmap(**heatmap_config)
