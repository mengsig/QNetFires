#global imports
import os
import sys
import shutil
import numpy as np
from  pprint import pprint
#pyretechnics imports
from pyretechnics.space_time_cube import SpaceTimeCube
import pyretechnics.eulerian_level_set as els
import time


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

# -------------Tunable parameters------------- #
xlen,ylen, savename, centrality, fuel_break_fraction = parse_args()
time_steps = int(1500 * np.sqrt(xlen*ylen)/(400))
# ----------end of tunable parameters----------#


savedir = f"results/{savename}"
fuel_breaks_file = f"{savedir}/{centrality}_{fuel_break_fraction}.txt"
fuel_breaks_img = f"{savedir}/{centrality}_{fuel_break_fraction}.png"
# copy the file into that directory (keeps the same basename)

#fuel_breaks_file = ""
if fuel_breaks_file != "":
    savedir = fuel_breaks_file.removesuffix(".txt")
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

# implementing fuel-breaks
if fuel_breaks_file != "":
    fuel_breaks = np.loadtxt(fuel_breaks_file).astype(bool)
    fuel_model[fuel_breaks] = 91
    cc[fuel_breaks] = 0
    cbd[fuel_breaks] = 0
#    ch[fuel_breaks] = 0
#    cbh[fuel_breaks] = 0

rel_img_name = fuel_breaks_img.split("/")[-1]
#clean repo
shutil.move(fuel_breaks_img, savedir + f"/{rel_img_name}")
os.remove(fuel_breaks_file)

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
    "fuel_moisture_live_herbaceous": SpaceTimeCube(cube_shape, 0.90),  # kg moisture/kg ovendry weight
    "fuel_moisture_live_woody"     : SpaceTimeCube(cube_shape, 0.60),  # kg moisture/kg ovendry weight
    "foliar_moisture"              : SpaceTimeCube(cube_shape, 0.90),
    "fuel_spread_adjustment"       : SpaceTimeCube(cube_shape, 1.0),
    "weather_spread_adjustment"    : SpaceTimeCube(cube_shape, 1.0),
}

# Day 2 @ 10:30am
start_time = 0# minutes

# 8 hours
max_duration = int(time_steps*2/3) * 60 # minutes

#ignite in the middle 
xcord, ycord = int(xlen/2), int(ylen/2)
num_burned_cells = 0
burned_cells_threshold = (xlen*ylen)/np.sqrt(xlen*ylen)

acres_burned = 0
simulation_runtime = 0
runtime_per_burned_cell = 0

while num_burned_cells < burned_cells_threshold:
    xcord += np.random.choice([-1,1])
    ycord += np.random.choice([-1,1])
    spread_state = els.SpreadState(cube_shape).ignite_cell((xcord,ycord))
    cube_resolution = (
        60, # band_duration: minutes
        30, # cell_height:   meters
        30, # cell_width:    meters
    )

#============================================================================================
# Spread fire from the start time for the max duration
#============================================================================================

    runtime_start       = time.perf_counter()
    fire_spread_results = els.spread_fire_with_phi_field(space_time_cubes,
                                                         spread_state,
                                                         cube_resolution,
                                                         start_time,
                                                         max_duration,
                                                         surface_lw_ratio_model="behave")
    runtime_stop        = time.perf_counter();
    stop_time           = fire_spread_results["stop_time"]      # minutes
    stop_condition      = fire_spread_results["stop_condition"] # "max duration reached" or "no burnable cells"
    spread_state        = fire_spread_results["spread_state"]   # updated SpreadState object (mutated from inputs)
    output_matrices     = spread_state.get_full_matrices()

#============================================================================================
# Print out the acres burned, total runtime, and runtime per burned cell
#============================================================================================

    num_burned_cells        = np.count_nonzero(output_matrices["fire_type"]) # cells
    if num_burned_cells == 0:
        continue
    acres_burned            = num_burned_cells / 4.5                         # acres
    simulation_runtime      = runtime_stop - runtime_start                   # seconds
    runtime_per_burned_cell = 1000.0 * simulation_runtime / num_burned_cells # ms/cell

print(f"[SIMULATE-{centrality}]: Acres Burned: " + str(acres_burned))
print(f"[SIMULATE-{centrality}]: Total Runtime: " + str(simulation_runtime) + " seconds")
print(f"[SIMULATE-{centrality}]: Runtime Per Burned Cell: " + str(runtime_per_burned_cell) + " ms/cell")

#============================================================================================
# Display summary statistics of our fire spread results
#============================================================================================

# Used as a filter in get_array_stats below
burned_cells = output_matrices["fire_type"] > 0

def get_array_stats(array, use_burn_scar_mask=True):
    array_values_to_analyze = array[burned_cells] if use_burn_scar_mask else array
    if len(array_values_to_analyze) > 0:
        return {
            "Min"  : np.min(array_values_to_analyze),
            "Max"  : np.max(array_values_to_analyze),
            "Mean" : np.mean(array_values_to_analyze),
            "Stdev": np.std(array_values_to_analyze),
        }
    else:
        return {
            "Min"  : "No Data",
            "Max"  : "No Data",
            "Mean" : "No Data",
            "Stdev": "No Data",
        }


#----------------- Saving statistics -----------------
vmin, vmax = output_matrices["fireline_intensity"].min() + 1, output_matrices["fireline_intensity"].max()
from contextlib import redirect_stdout
out_path = f"{savedir}/stats.txt"
with open(out_path, "w") as fout, redirect_stdout(fout):
    print("Fire Behavior from Day 2 @ 10:30am - Day 2 @ 6:30pm Spreading from Coordinate (50,50)\n" + "=" * 100)

    print("Stop Time: " + str(stop_time) + " (minutes)")
    print("Stop Condition: " + stop_condition)
    print(f"Acres Burned: " + str(acres_burned))
    print(f"Total Runtime: " + str(simulation_runtime) + " seconds")
    print(f" Runtime Per Burned Cell: " + str(runtime_per_burned_cell) + " ms/cell")


    print("\nPhi (phi <= 0: burned, phi > 0: unburned")
    pprint(get_array_stats(output_matrices["phi"], use_burn_scar_mask=False), sort_dicts=False)

    print("\nFire Type (0=unburned, 1=surface, 2=passive_crown, 3=active_crown)")
    pprint(get_array_stats(output_matrices["fire_type"]), sort_dicts=False)

    print("\nSpread Rate (m/min)")
    pprint(get_array_stats(output_matrices["spread_rate"]), sort_dicts=False)

    print("\nSpread Direction (degrees clockwise from North)")
    pprint(get_array_stats(output_matrices["spread_direction"]), sort_dicts=False)

    print("\nFireline Intensity (kW/m)")
    pprint(get_array_stats(output_matrices["fireline_intensity"]), sort_dicts=False)

    print("\nFlame Length (meters)")
    pprint(get_array_stats(output_matrices["flame_length"]), sort_dicts=False)

    print("\nTime of Arrival (minutes)")
    pprint(get_array_stats(output_matrices["time_of_arrival"]), sort_dicts=False)

#------------ Finished saving statistics ------------
import matplotlib.pyplot as plt
import numpy as np

output_matrices["fireline_intensity"] += 0.01
# See https://matplotlib.org/stable/gallery/color/colormap_reference.html for the available options for "colors"
heatmap_configs = [
    {
        "matrix"  : output_matrices["phi"],
        "colors"  : "plasma",
        "units"   : "phi <= 0: burned, phi > 0: unburned",
        "title"   : "Phi",
        "filename": f"{savedir}/els_phi.png",
    },
    {
        "matrix"  : output_matrices["fire_type"],
        "colors"  : "viridis",
        "units"   : "0=unburned, 1=surface, 2=passive_crown, 3=active_crown",
        "title"   : "Fire Type",
        "filename": f"{savedir}/els_fire_type.png",
        "vmin"    : 0,
        "vmax"    : 3,
        "ticks"   : [0,1,2,3],
    },
    {
        "matrix"  : output_matrices["spread_rate"],
        "colors"  : "hot",
        "units"   : "m/min",
        "title"   : "Spread Rate",
        "filename": f"{savedir}/els_spread_rate.png",
    },
    {
        "matrix"  : output_matrices["spread_direction"],
        "colors"  : "viridis",
        "units"   : "degrees clockwise from North",
        "title"   : "Spread Direction",
        "filename": f"{savedir}/els_spread_direction.png",
        "vmin"    : 0,
        "vmax"    : 360,
        "ticks"   : [0,45,90,135,180,225,270,315,360]
    },
    {
        "matrix"  : output_matrices["fireline_intensity"],
        "colors"  : "hot",
        "units"   : "kW/m",
        "title"   : "Fireline Intensity",
        "filename": f"{savedir}/els_fireline_intensity.png",
        "vmin"    : output_matrices['fireline_intensity'].min(),
        "vmax"    : output_matrices['fireline_intensity'].max(),
        "norm"    : True
    },
    {
        "matrix"  : output_matrices["flame_length"],
        "colors"  : "hot",
        "units"   : "meters",
        "title"   : "Flame Length",
        "filename": f"{savedir}/els_flame_length.png",
    },
    {
            "matrix"  : np.flip(cc_cube[0,:,:],axis=0),
        "colors"  : "Greens",
        "units"   : "coverage",
        "title"   : "canopy coverage",
        "filename": f"{savedir}/els_canopy_coverage.png",
    },
    {
        "matrix"  : output_matrices["time_of_arrival"],
        "colors"  : "viridis",
        "units"   : "minutes",
        "vmin"    : 1,
        "vmax"    : stop_time,
        "title"   : "Time of Arrival",
        "filename": f"{savedir}/els_time_of_arrival.png",
    },
]

output_matrices["phi"][0:xlen, 0:ylen][fuel_breaks] = np.nan
output_matrices["fire_type"][0:xlen, 0:ylen][fuel_breaks] = 0
output_matrices["spread_rate"][0:xlen, 0:ylen][fuel_breaks] = np.nan
output_matrices["spread_direction"][0:xlen, 0:ylen][fuel_breaks] = np.nan
output_matrices["fireline_intensity"][0:xlen, 0:ylen][fuel_breaks] = np.inf
output_matrices["flame_length"][0:xlen, 0:ylen][fuel_breaks] = np.nan
output_matrices["time_of_arrival"][0:xlen, 0:ylen][fuel_breaks] = np.nan

#contour_configs = []


for heatmap_config in heatmap_configs:
    save_matrix_as_heatmap(**heatmap_config)


#for contour_config in contour_configs:
#    save_matrix_as_contours(**contour_config)


import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# --- parameters for the GIF ---
sample_interval = 200                # minutes between frames
out_dir     = f"{savedir}gif_frames"
gif_name    = f"{savedir}/fireline_intensity.gif"
duration    = 0.5                   # seconds per frame
# --- end of gif parameters ---

#make sure output directory exists
os.makedirs(out_dir, exist_ok=True)

#extract final matrices
fi  = output_matrices["fireline_intensity"]   # kW/m
toa = output_matrices["time_of_arrival"]      # minutes since ignition

#generate list of times (0 .. stop_time) at your interval
times = np.arange(0, int(stop_time) + 1, sample_interval)

frames = []
cc = np.flip(cc_cube[0,:,:], axis = 0)
cc[fuel_breaks] = np.inf
cmap = plt.get_cmap('hot')
cmap.set_bad("purple")
cmap.set_under(alpha=0)

cmap_green = plt.get_cmap("Greens")
cmap_green.set_bad("purple")
cmap_green.set_under(alpha=0)
norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
for t in times:
    mask = (toa > 0) & (toa <= t)

    # fire in hot:
    fire_frame = np.where(mask, fi, -10)
    if fire_frame.max() < 1:
        fire_frame += 1-fire_frame.max()

    # static canopy (2D)
    canopy_frame = np.where(~mask, cc, -10)
    canopy_frame[fuel_breaks] = np.inf

    fig, ax = plt.subplots()
    ax.imshow(canopy_frame,
              origin="lower",
              cmap=cmap_green,
              vmin=0,
              vmax=1,
              interpolation = "nearest",
              aspect = "equal")
    # 2) draw fire and capture the mappable
    im = ax.imshow(fire_frame,
                   origin="lower",
                   cmap=cmap,
                   norm=norm,          # reuse the sane LogNorm
                   alpha=0.6,
                   interpolation="nearest",
                   aspect="equal")
    ax.set_title(f"t = {t} min")
    ax.axis("off")

    # 3) only fire colorbar
    cbar = fig.colorbar(im, ax=ax,
                        fraction=0.046, pad=0.04,
                        label="Fireline Intensity (kW/m)")

    # save frame
    fname = os.path.join(out_dir, f"frame_{t:04d}.png")
    fig.savefig(fname, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    frames.append(imageio.v2.imread(fname))

import shutil
if os.path.isdir(out_dir):
    shutil.rmtree(out_dir)
    print(f"[SIMULATE-{centrality}]: Deleted temp directory {out_dir}")
else:
    print(f"[SIMULATE-{centrality}]: No such directory: {out_dir}")
#save the GIF
imageio.mimsave(gif_name, frames, duration=duration, loop = 0)
print(f"[SIMULATE-{centrality}]: → Saved GIF: {gif_name}")
