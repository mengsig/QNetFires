#ntglobal imports
import os
import sys
import numpy as np
#pyretechnics import
from pyretechnics.space_time_cube import SpaceTimeCube
import pyretechnics.burn_cells as bc
from DomiRank import domirank
from scipy.sparse.linalg import eigs

#src/utils/ imports
script_dir   = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.insert(0, project_root)
from src.utils.loadingUtils import (
    load_all_rasters,
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
    create_network,
    )

def count_files(dir_path):
    # list all entries and filter to files only
    return sum(
        1 for name in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, name))
    )

basename = "cropped_raster"
name = f"{basename}/domirank"
print(f"[CREATE-ADJACENCY]: Saving to {name}...")
fireline_name = f"{basename}/fireline"
os.makedirs(name, exist_ok = True)
os.makedirs(f"{fireline_name}", exist_ok = True)

files = count_files("cropped_raster/slp")

for k in range(files):
# -------------Tunable parameters------------- #
    time_steps = 1
# ----------end of tunable parameters----------#


#directory containing the already-cropped rasters
    raster_dir = "cropped_rasters"

    rasterDict = load_all_rasters("cropped_raster", k)
    slope, aspect, dem, cc, cbd, cbh, ch, fuel_model = rasterDict.values()
    xlen, ylen = slope.shape

#convert to cubes
# Define cube shape (e.g. 24 hours)
    rows, cols = slope.shape
    cube_shape = (time_steps, rows, cols)

#build space-time cubes
    space_time_cubes = {
        "slope"                        : SpaceTimeCube(cube_shape, slope),
        "aspect"                       : SpaceTimeCube(cube_shape, aspect),
        "fuel_model"                   : SpaceTimeCube(cube_shape, fuel_model),
        "canopy_cover"                 : SpaceTimeCube(cube_shape, cc),
        "canopy_height"                : SpaceTimeCube(cube_shape, ch),
        "canopy_base_height"           : SpaceTimeCube(cube_shape, cbh),
        "canopy_bulk_density"          : SpaceTimeCube(cube_shape, cbd),
        "wind_speed_10m"               : SpaceTimeCube(cube_shape, 0),
        "upwind_direction"             : SpaceTimeCube(cube_shape, 0),
        "fuel_moisture_dead_1hr"       : SpaceTimeCube(cube_shape, 0.05),
        "fuel_moisture_dead_10hr"      : SpaceTimeCube(cube_shape, 0.10),
        "fuel_moisture_dead_100hr"     : SpaceTimeCube(cube_shape, 0.25),
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
        print(f"[CREATE-ADJACENCY]: direction {i}/{directions}")
        num_simulations += 1
#============================================================================================
# Calculate combined fire behavior in the direction of the azimuth (with wind limit)
#============================================================================================

        combined_behavior_limited = bc.burn_all_cells_toward_azimuth(space_time_cubes,
                                                                     spread_azimuth,
                                                                     0,
                                                                     y_range,
                                                                     x_range,
                                                                     surface_lw_ratio_model="behave")

#============================================================================================
# TODO: do we use unlimited or limited?
#============================================================================================
        spread_rate_mean[i] += combined_behavior_limited["fireline_intensity"]

#============================================================================================
# Update spread azimuth angle
#============================================================================================
        spread_azimuth += azimuth_step

        #normalize spread_rate_mean
        spread_rate_mean[i,::,::] /= num_simulations

#============================================================================================
    np.savetxt(f"{fireline_name}/fireline_north_{k}.txt", spread_rate_mean[0])
    np.savetxt(f"{fireline_name}/fireline_east_{k}.txt", spread_rate_mean[1])
    np.savetxt(f"{fireline_name}/fireline_south_{k}.txt", spread_rate_mean[2])
    np.savetxt(f"{fireline_name}/fireline_west_{k}.txt", spread_rate_mean[3])
    edgelist = build_edgelist_from_spread_rates(spread_rate_mean, x, y)
    edgelist_array = np.ascontiguousarray(edgelist, dtype=np.float32)

    G = create_network(edgelist_array, sparse_array = True)
    G /= G.max() # normalization
    degree = G.sum(axis = 0)
    degree += 1 # for plotting
    plot_degree = np.reshape(degree, (xlen, ylen))
    vmin = degree.min()
    vmax = degree.max()

# Extracting the centrality measures based on the user input
    lambN, _ = eigs(G, k = 1, which = "SR")
    sigma = 1 - 1/(G.shape[0])
    sigma = 0.999
    print(f"[GENERATING-FUEL-BREAKS-domirank:] using sigma {sigma}...")
    _, centralityDistribution = domirank(G, sigma = -sigma/lambN, analytical = True) 
    centralityDistribution = centralityDistribution.real

    np.savetxt(f"{name}/domirank_{k}.txt", centralityDistribution)
