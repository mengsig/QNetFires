import numpy as np
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import os

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
        
        # Parallel processing configuration - more conservative
        # Check if we're likely in a multiprocessing environment already
        self.max_workers_per_env = 1  # Conservative default
        self.use_parallel_simulations = False  # Disable by default to avoid over-parallelization
        
        print(f"🔥 Simulate initialized with {self.max_workers_per_env} workers per environment (parallel: {self.use_parallel_simulations})")

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
            "fuel_moisture_dead_1hr"       : SpaceTimeCube(self.cube_shape, np.float32(0.05)),
            "fuel_moisture_dead_10hr"      : SpaceTimeCube(self.cube_shape, np.float32(0.10)),
            "fuel_moisture_dead_100hr"     : SpaceTimeCube(self.cube_shape, np.float32(0.25)),
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
                "fuel_moisture_dead_1hr"       : SpaceTimeCube(self.cube_shape, np.float32(0.05)),
                "fuel_moisture_dead_10hr"      : SpaceTimeCube(self.cube_shape, np.float32(0.10)),
                "fuel_moisture_dead_100hr"     : SpaceTimeCube(self.cube_shape, np.float32(0.25)),
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

    def run_many_simulations(self, num_simulations, max_duration=None):
        """
        Run multiple fire simulations with random ignition points.
        Now with balanced parallelization to avoid over-parallelization.
        
        Args:
            num_simulations: Number of simulations to run
            max_duration: Maximum duration for each simulation (minutes)
        """
        start_time = time.time()
        
        # Generate random ignition points
        ignition_points = [
            (np.random.randint(0, self.slope.shape[0]), np.random.randint(0, self.slope.shape[1]))
            for _ in range(num_simulations)
        ]
        
        # Use parallel simulations only if specifically enabled and beneficial
        if self.use_parallel_simulations and num_simulations > 1 and self.max_workers_per_env > 1:
            # PARALLEL EXECUTION - Use with caution to avoid over-parallelization
            burned_matrices = []
            total_acres_burned = 0
            
            # Use ProcessPoolExecutor for CPU-intensive tasks
            with ProcessPoolExecutor(max_workers=self.max_workers_per_env) as executor:
                # Submit all simulations
                future_to_point = {}
                for i, (xcord, ycord) in enumerate(ignition_points):
                    future = executor.submit(
                        _run_single_simulation_worker,
                        self._get_simulation_params(),
                        xcord, ycord, max_duration
                    )
                    future_to_point[future] = (i, xcord, ycord)
                
                # Collect results as they complete
                for future in as_completed(future_to_point):
                    i, xcord, ycord = future_to_point[future]
                    try:
                        result = future.result()
                        burned_matrices.append(result['burned_matrix'])
                        total_acres_burned += result['acres_burned']
                    except Exception as e:
                        print(f"Simulation {i} failed: {e}")
                        # Create empty result for failed simulation
                        burned_matrices.append(np.zeros_like(self.slope, dtype=np.uint8))
            
            # Combine results
            if burned_matrices:
                self.burned = burned_matrices[0]
                for burned_matrix in burned_matrices[1:]:
                    self.burned += burned_matrix
                self.average_acres_burned = total_acres_burned
            else:
                self.burned = np.zeros_like(self.slope, dtype=np.uint8)
                self.average_acres_burned = 0
                
        else:
            # SEQUENTIAL EXECUTION - More efficient when already in parallel environments
            for i, (xcord, ycord) in enumerate(ignition_points):
                if max_duration is not None:
                    self.run_simulation_with_duration(xcord, ycord, max_duration)
                else:
                    self.run_simulation(xcord, ycord)
                    
                if i == 0:
                    self.burned = self.output_matrices["fire_type"].astype(np.uint8)
                else:
                    self.burned += self.output_matrices["fire_type"].astype(np.uint8)
                self.average_acres_burned += self.acres_burned
        
        elapsed_time = time.time() - start_time
        # Only print timing for longer simulations to reduce output noise
        if elapsed_time > 2.0:
            print(f"   🔥 Completed {num_simulations} simulations in {elapsed_time:.2f}s "
                  f"({num_simulations/elapsed_time:.1f} sims/sec)")
    
    def _get_simulation_params(self):
        """Get parameters needed for worker processes."""
        return {
            'slope': self.slope,
            'aspect': self.aspect,
            'dem': self.dem,
            'cc': self.cc,
            'cbd': self.cbd,
            'cbh': self.cbh,
            'ch': self.ch,
            'fuel_model': self.fuel_model.copy(),  # Copy to avoid race conditions
            'cube_shape': self.cube_shape,
            'fuel_breaks': self.fuel_breaks
        }
    
    def set_parallel_simulations(self, enabled: bool, max_workers: int = None):
        """
        Enable/disable parallel simulations and set worker count.
        
        Args:
            enabled: Whether to use parallel simulations
            max_workers: Maximum number of worker processes (None for auto)
        """
        self.use_parallel_simulations = enabled
        if max_workers is not None:
            self.max_workers_per_env = max_workers
        
        print(f"🔥 Parallel simulations {'enabled' if enabled else 'disabled'}, "
              f"max_workers: {self.max_workers_per_env}")
                
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


def _run_single_simulation_worker(params, xcord, ycord, max_duration=None):
    """
    Worker function for parallel simulation execution.
    This runs in a separate process.
    """
    import pyretechnics.eulerian_level_set as els
    from pyretechnics.space_time_cube import SpaceTimeCube
    
    # Recreate space-time cubes in worker process
    slope_f32 = params['slope'].astype(np.float32)
    aspect_f32 = params['aspect'].astype(np.float32)
    fuel_model_f32 = params['fuel_model'].astype(np.float32)
    cc_f32 = params['cc'].astype(np.float32)
    ch_f32 = params['ch'].astype(np.float32)
    cbh_f32 = params['cbh'].astype(np.float32)
    cbd_f32 = params['cbd'].astype(np.float32)
    
    # Apply fuel breaks if present
    if params['fuel_breaks'] is not None:
        fuel_model_f32[params['fuel_breaks']] = 91  # Set fuel model to 91 (non-burnable)
    
    cube_shape = params['cube_shape']
    space_time_cubes = {
        "slope"                        : SpaceTimeCube(cube_shape, slope_f32),
        "aspect"                       : SpaceTimeCube(cube_shape, aspect_f32),
        "fuel_model"                   : SpaceTimeCube(cube_shape, fuel_model_f32),
        "canopy_cover"                 : SpaceTimeCube(cube_shape, cc_f32),
        "canopy_height"                : SpaceTimeCube(cube_shape, ch_f32),
        "canopy_base_height"           : SpaceTimeCube(cube_shape, cbh_f32),
        "canopy_bulk_density"          : SpaceTimeCube(cube_shape, cbd_f32),
        "wind_speed_10m"               : SpaceTimeCube(cube_shape, np.float32(0)),
        "upwind_direction"             : SpaceTimeCube(cube_shape, np.float32(0)),
        "fuel_moisture_dead_1hr"       : SpaceTimeCube(cube_shape, np.float32(0.05)),
        "fuel_moisture_dead_10hr"      : SpaceTimeCube(cube_shape, np.float32(0.10)),
        "fuel_moisture_dead_100hr"     : SpaceTimeCube(cube_shape, np.float32(0.25)),
        "fuel_moisture_live_herbaceous": SpaceTimeCube(cube_shape, np.float32(0.90)),
        "fuel_moisture_live_woody"     : SpaceTimeCube(cube_shape, np.float32(0.60)),
        "foliar_moisture"              : SpaceTimeCube(cube_shape, np.float32(0.90)),
        "fuel_spread_adjustment"       : SpaceTimeCube(cube_shape, np.float32(1.0)),
        "weather_spread_adjustment"    : SpaceTimeCube(cube_shape, np.float32(1.0)),
    }
    
    # Run simulation
    spread_state = els.SpreadState(cube_shape).ignite_cell((xcord, ycord))
    cube_resolution = (60, 30, 30)  # minutes, meters, meters
    
    if max_duration is not None:
        sim_max_duration = max_duration
    else:
        sim_max_duration = int((cube_shape[0] * 2 / 3) * 60)
    
    fire_spread_results = els.spread_fire_with_phi_field(
        space_time_cubes,
        spread_state,
        cube_resolution,
        0,
        sim_max_duration,
        surface_lw_ratio_model="behave"
    )
    
    spread_state = fire_spread_results["spread_state"]
    output_matrices = spread_state.get_full_matrices()
    
    num_burned_cells = np.count_nonzero(output_matrices["fire_type"])
    burned_matrix = output_matrices["fire_type"].astype(np.uint8)
    
    return {
        'burned_matrix': burned_matrix,
        'acres_burned': num_burned_cells
    }
