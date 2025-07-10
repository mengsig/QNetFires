import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

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
        self._simulation_lock = threading.Lock()  # For thread-safe operations

    def set_space_time_cubes(self, time_steps = None):
        from pyretechnics.space_time_cube import SpaceTimeCube
        if time_steps is None:
            time_steps = int(1500 * np.sqrt(self.slope.shape[0] * self.slope.shape[1]) / 400)
        self.cube_shape = (time_steps, self.slope.shape[0], self.slope.shape[1])
        
        # Convert all arrays to float32 to suppress warnings
        slope_f32 = self.slope.astype(np.float32)
        aspect_f32 = self.aspect.astype(np.float32)
        fuel_model_f32 = self.fuel_model.astype(np.float32)
        cc_f32 = self.cc.astype(np.float32)
        ch_f32 = self.ch.astype(np.float32)
        cbh_f32 = self.cbh.astype(np.float32)
        cbd_f32 = self.cbd.astype(np.float32)
        
        self.space_time_cubes = {
            "slope"                        : SpaceTimeCube(self.cube_shape, slope_f32),
            "aspect"                       : SpaceTimeCube(self.cube_shape, aspect_f32),
            "fuel_model"                   : SpaceTimeCube(self.cube_shape, fuel_model_f32),
            "canopy_cover"                 : SpaceTimeCube(self.cube_shape, cc_f32),
            "canopy_height"                : SpaceTimeCube(self.cube_shape, ch_f32),
            "canopy_base_height"           : SpaceTimeCube(self.cube_shape, cbh_f32),
            "canopy_bulk_density"          : SpaceTimeCube(self.cube_shape, cbd_f32),
            "wind_speed_10m"               : SpaceTimeCube(self.cube_shape, np.float32(0)),
            "upwind_direction"             : SpaceTimeCube(self.cube_shape, np.float32(0)),
            "fuel_moisture_dead_1hr"       : SpaceTimeCube(self.cube_shape, np.float32(0.10)),
            "fuel_moisture_dead_10hr"      : SpaceTimeCube(self.cube_shape, np.float32(0.25)),
            "fuel_moisture_dead_100hr"     : SpaceTimeCube(self.cube_shape, np.float32(0.50)),
            "fuel_moisture_live_herbaceous": SpaceTimeCube(self.cube_shape, np.float32(0.90)),  # kg moisture/kg ovendry weight
            "fuel_moisture_live_woody"     : SpaceTimeCube(self.cube_shape, np.float32(0.60)),  # kg moisture/kg ovendry weight
            "foliar_moisture"              : SpaceTimeCube(self.cube_shape, np.float32(0.90)),
            "fuel_spread_adjustment"       : SpaceTimeCube(self.cube_shape, np.float32(1.0)),
            "weather_spread_adjustment"    : SpaceTimeCube(self.cube_shape, np.float32(1.0)),
        }

    def set_fuel_breaks(self, fuel_breaks = None):
        """
        Apply fuel breaks to the fuel model and canopy cover.
        """
        from pyretechnics.space_time_cube import SpaceTimeCube
        self.fuel_breaks = fuel_breaks
        if fuel_breaks is not None:
            self.fuel_model[fuel_breaks] = 91  # Set fuel model to 91 (non-burnable)
            
            # Convert all arrays to float32 to suppress warnings
            slope_f32 = self.slope.astype(np.float32)
            aspect_f32 = self.aspect.astype(np.float32)
            fuel_model_f32 = self.fuel_model.astype(np.float32)
            cc_f32 = self.cc.astype(np.float32)
            ch_f32 = self.ch.astype(np.float32)
            cbh_f32 = self.cbh.astype(np.float32)
            cbd_f32 = self.cbd.astype(np.float32)
            
            self.space_time_cubes = {
                "slope"                        : SpaceTimeCube(self.cube_shape, slope_f32),
                "aspect"                       : SpaceTimeCube(self.cube_shape, aspect_f32),
                "fuel_model"                   : SpaceTimeCube(self.cube_shape, fuel_model_f32),
                "canopy_cover"                 : SpaceTimeCube(self.cube_shape, cc_f32),
                "canopy_height"                : SpaceTimeCube(self.cube_shape, ch_f32),
                "canopy_base_height"           : SpaceTimeCube(self.cube_shape, cbh_f32),
                "canopy_bulk_density"          : SpaceTimeCube(self.cube_shape, cbd_f32),
                "wind_speed_10m"               : SpaceTimeCube(self.cube_shape, np.float32(0)),
                "upwind_direction"             : SpaceTimeCube(self.cube_shape, np.float32(0)),
                "fuel_moisture_dead_1hr"       : SpaceTimeCube(self.cube_shape, np.float32(0.10)),
                "fuel_moisture_dead_10hr"      : SpaceTimeCube(self.cube_shape, np.float32(0.25)),
                "fuel_moisture_dead_100hr"     : SpaceTimeCube(self.cube_shape, np.float32(0.50)),
                "fuel_moisture_live_herbaceous": SpaceTimeCube(self.cube_shape, np.float32(0.90)),  # kg moisture/kg ovendry weight
                "fuel_moisture_live_woody"     : SpaceTimeCube(self.cube_shape, np.float32(0.60)),  # kg moisture/kg ovendry weight
                "foliar_moisture"              : SpaceTimeCube(self.cube_shape, np.float32(0.90)),
                "fuel_spread_adjustment"       : SpaceTimeCube(self.cube_shape, np.float32(1.0)),
                "weather_spread_adjustment"    : SpaceTimeCube(self.cube_shape, np.float32(1.0)),
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

    def _run_single_simulation(self, ignition_point, max_duration=None):
        """Run a single simulation and return the results."""
        xcord, ycord = ignition_point
        
        if max_duration is not None:
            self.run_simulation_with_duration(xcord, ycord, max_duration)
        else:
            self.run_simulation(xcord, ycord)
            
        return {
            'acres_burned': self.acres_burned,
            'burned_matrix': self.output_matrices["fire_type"].astype(np.uint8)
        }

    def run_many_simulations(self, num_simulations, max_duration=None, use_parallel=True, max_workers=None):
        """
        Run multiple fire simulations with random ignition points.
        
        Args:
            num_simulations: Number of simulations to run
            max_duration: Maximum duration for each simulation (minutes)
            use_parallel: Whether to run simulations in parallel (default: True)
            max_workers: Maximum number of parallel workers (default: min(4, num_simulations))
        """
        start_time = time.time()
        
        # Generate random ignition points
        ignition_points = []
        for _ in range(num_simulations):
            xcord = np.random.randint(0, self.slope.shape[0])
            ycord = np.random.randint(0, self.slope.shape[1])
            ignition_points.append((xcord, ycord))
        
        if use_parallel and num_simulations > 1:
            # Parallel execution
            max_workers = max_workers or min(4, num_simulations)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all simulation tasks
                futures = []
                for ignition_point in ignition_points:
                    future = executor.submit(self._run_single_simulation, ignition_point, max_duration)
                    futures.append(future)
                
                # Collect results
                results = []
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                
                # Aggregate results
                self.average_acres_burned = 0
                for i, result in enumerate(results):
                    if i == 0:
                        self.burned = result['burned_matrix']
                    else:
                        self.burned += result['burned_matrix']
                    self.average_acres_burned += result['acres_burned']
        else:
            # Sequential execution (original method)
            for i, ignition_point in enumerate(ignition_points):
                xcord, ycord = ignition_point
                
                if max_duration is not None:
                    self.run_simulation_with_duration(xcord, ycord, max_duration)
                else:
                    self.run_simulation(xcord, ycord)
                    
                if i == 0:
                    self.burned = self.output_matrices["fire_type"].astype(np.uint8)
                else:
                    self.burned += self.output_matrices["fire_type"].astype(np.uint8)
                self.average_acres_burned += self.acres_burned
        
        simulation_time = time.time() - start_time
        
        # Store final average
        self.acres_burned = self.average_acres_burned
        
        # Debug output
        if hasattr(self, '_debug_parallel') and self._debug_parallel:
            print(f"      Fire simulations: {num_simulations} sims in {simulation_time:.2f}s "
                  f"({'parallel' if use_parallel and num_simulations > 1 else 'sequential'})")
            
    def run_simulation_with_duration(self, xcord, ycord, max_duration_minutes):
        """Run simulation with specified maximum duration."""
        import pyretechnics.eulerian_level_set as els
        spread_state = els.SpreadState(self.cube_shape).ignite_cell((xcord,ycord))
        cube_resolution = (
            60, # band_duration: minutes
            30, # cell_height:   meters
            30, # cell_width:    meters
        )

        fire_spread_results = els.spread_fire_with_phi_field(self.space_time_cubes,
                                                             spread_state,
                                                             cube_resolution,
                                                             0,
                                                             max_duration_minutes,
                                                             surface_lw_ratio_model="behave")

        spread_state        = fire_spread_results["spread_state"]
        self.output_matrices = spread_state.get_full_matrices()

        num_burned_cells = np.count_nonzero(self.output_matrices["fire_type"])
        self.acres_burned = num_burned_cells 
        self.burned = self.output_matrices["fire_type"].astype(np.uint8)

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
