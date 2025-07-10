import numpy as np
class Simulate:
    def __init__(self, slope, aspect, dem, cc, cbd, cbh, ch, fuel_model):
        self.slope = slope
        self.aspect = aspect
        self.dem = dem
        self.cc = cc
        self.cbd = cbd
        self.cbh = cbh
        self.ch = ch
        self.fuel_model = fuel_model
        self.average_acres_burned = 0

    def set_space_time_cubes(self, time_steps = None):
        from pyretechnics.space_time_cube import SpaceTimeCube
        if time_steps is None:
            time_steps = int(1500 * np.sqrt(self.slope.shape[0] * self.slope.shape[1]) / 400)
        self.cube_shape = (time_steps, self.slope.shape[0], self.slope.shape[1])
        self.space_time_cubes = {
            "slope"                        : SpaceTimeCube(self.cube_shape, self.slope),
            "aspect"                       : SpaceTimeCube(self.cube_shape, self.aspect),
            "fuel_model"                   : SpaceTimeCube(self.cube_shape, self.fuel_model),
            "canopy_cover"                 : SpaceTimeCube(self.cube_shape, self.cc),
            "canopy_height"                : SpaceTimeCube(self.cube_shape, self.ch),
            "canopy_base_height"           : SpaceTimeCube(self.cube_shape, self.cbh),
            "canopy_bulk_density"          : SpaceTimeCube(self.cube_shape, self.cbd),
            "wind_speed_10m"               : SpaceTimeCube(self.cube_shape, 0),
            "upwind_direction"             : SpaceTimeCube(self.cube_shape, 0),
            "fuel_moisture_dead_1hr"       : SpaceTimeCube(self.cube_shape, 0.10),
            "fuel_moisture_dead_10hr"      : SpaceTimeCube(self.cube_shape, 0.25),
            "fuel_moisture_dead_100hr"     : SpaceTimeCube(self.cube_shape, 0.50),
            "fuel_moisture_live_herbaceous": SpaceTimeCube(self.cube_shape, 0.90),  # kg moisture/kg ovendry weight
            "fuel_moisture_live_woody"     : SpaceTimeCube(self.cube_shape, 0.60),  # kg moisture/kg ovendry weight
            "foliar_moisture"              : SpaceTimeCube(self.cube_shape, 0.90),
            "fuel_spread_adjustment"       : SpaceTimeCube(self.cube_shape, 1.0),
            "weather_spread_adjustment"    : SpaceTimeCube(self.cube_shape, 1.0),
        }

    def set_fuel_breaks(self, fuel_breaks = None):
        """
        Apply fuel breaks to the fuel model and canopy cover.
        """
        from pyretechnics.space_time_cube import SpaceTimeCube
        self.fuel_breaks = fuel_breaks
        if fuel_breaks is not None:
            self.fuel_model[fuel_breaks] = 91  # Set fuel model to 91 (non-burnable)
            self.space_time_cubes = {
                "slope"                        : SpaceTimeCube(self.cube_shape, self.slope),
                "aspect"                       : SpaceTimeCube(self.cube_shape, self.aspect),
                "fuel_model"                   : SpaceTimeCube(self.cube_shape, self.fuel_model),
                "canopy_cover"                 : SpaceTimeCube(self.cube_shape, self.cc),
                "canopy_height"                : SpaceTimeCube(self.cube_shape, self.ch),
                "canopy_base_height"           : SpaceTimeCube(self.cube_shape, self.cbh),
                "canopy_bulk_density"          : SpaceTimeCube(self.cube_shape, self.cbd),
                "wind_speed_10m"               : SpaceTimeCube(self.cube_shape, 0),
                "upwind_direction"             : SpaceTimeCube(self.cube_shape, 0),
                "fuel_moisture_dead_1hr"       : SpaceTimeCube(self.cube_shape, 0.10),
                "fuel_moisture_dead_10hr"      : SpaceTimeCube(self.cube_shape, 0.25),
                "fuel_moisture_dead_100hr"     : SpaceTimeCube(self.cube_shape, 0.50),
                "fuel_moisture_live_herbaceous": SpaceTimeCube(self.cube_shape, 0.90),  # kg moisture/kg ovendry weight
                "fuel_moisture_live_woody"     : SpaceTimeCube(self.cube_shape, 0.60),  # kg moisture/kg ovendry weight
                "foliar_moisture"              : SpaceTimeCube(self.cube_shape, 0.90),
                "fuel_spread_adjustment"       : SpaceTimeCube(self.cube_shape, 1.0),
                "weather_spread_adjustment"    : SpaceTimeCube(self.cube_shape, 1.0),
            }

    def run_simulation(self, xcord, ycord):
        import pyretechnics.eulerian_level_set as els
        spread_state = els.SpreadState(self.cube_shape).ignite_cell((xcord,ycord))
        cube_resolution = (
            60, # band_duration: minutes
            30, # cell_height:   meters
            30, # cell_width:    meters
        )

#============================================================================================
# Spread fire from the start time for the max duration
#============================================================================================
        max_duration = int((self.cube_shape[0] * 2 / 3) * 60) 

        fire_spread_results = els.spread_fire_with_phi_field(self.space_time_cubes,
                                                             spread_state,
                                                             cube_resolution,
                                                             0,
                                                             max_duration,
                                                             surface_lw_ratio_model="behave")

        spread_state        = fire_spread_results["spread_state"]   # updated SpreadState object (mutated from inputs)
        self.output_matrices     = spread_state.get_full_matrices()

#============================================================================================
# Print out the acres burned, total runtime, and runtime per burned cell
#============================================================================================

        num_burned_cells = np.count_nonzero(self.output_matrices["fire_type"]) # cells
        self.acres_burned = num_burned_cells 
        self.burned = self.output_matrices["fire_type"].astype(np.uint8)

    def run_many_simulations(self, num_simulations):
        for i in range(num_simulations):
            xcord = np.random.randint(0, self.slope.shape[0])
            ycord = np.random.randint(0, self.slope.shape[1])
            self.run_simulation(xcord, ycord)
            if i == 0:
                self.burned = self.output_matrices["fire_type"].astype(np.uint8)
            else:
                self.burned += self.output_matrices["fire_type"].astype(np.uint8)
            self.average_acres_burned += self.acres_burned

    def get_burned(self):
        """
        Get the burned acres from the last simulation.
        """
        return self.burned

    def get_loss(self):
        """
        Calculate the loss based on the acres burned.
        This is a placeholder for actual loss calculation logic.
        """
        # Placeholder logic for loss calculation
        if self.average_acres_burned == 0:
            return self.acres_burned 
        else:
            return self.average_acres_burned
