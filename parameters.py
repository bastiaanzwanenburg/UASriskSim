'''
Specifies all parameters used for a simulation run. Possible to set fixed_params, which are always used, and variable_params, which takes a list per parameter, and subsequently runs the model for each parameter in the list.
'''

from helpers.metrics import *
from env.loadEnvironment import loadEnvironment
import random
import numpy as np

x = []


scaling_factor = 5
risk_map_resolution = 5
n_iterations = 1  #Number of batch runs
max_steps = 1000 #Max steps for each batch run
n_agents = 1

MP = False

saveRiskMap = False

# Generate a new line
path_database = "database/" + str(random.random())
path_lockfile = "database/" + str(random.random())


fixed_params = {
    "city": "Paris",
    "scaling_factor": scaling_factor,
    "risk_map_resolution": risk_map_resolution,

    "path_database": path_database,
    "path_lockfile": path_lockfile,

    # "Hub_location": all_hub_locations,  # relative location on map, (0,0) on top-left, width 200, height 165
    # "Delivery_location": all_delivery_locations,

    # Either fixed hub/DP locations or random locations should be used
    # "Hub_location": [(72, 79)],
    # "Delivery_location": [(90, 75)],
    "n_random_hubs": 4,
    "n_random_delivery_locations": 100,


    "v_impact_scenario": 1, # 1 = use modeled impact speed, 2 = use terminal speed, 3 = use terminal speed + 40%
    "FailureRateCruise": 0.0005,  # P(failure) in one second
    "FailureRateClimbDescend": 0.0005, # P(failure) in one second
    "Risk_Optimal": "shelter_and_pop_density",  # Options: calculate_risk, shelter_and_pop_density, pop_density, False
    "Nofly": False, # Whether or not to use no-fly zones
    "conflictdetection": False, # Wheter or not to use CD&R (not implemented now)
    "total_demand_per_hour": 976,  # Demand in packages per hour
    "max_steps": max_steps, # Max simulation steps to take
    "random_hub_location": "edges", # Specifies where hub locations are placed
    # "scale_demand_with_density": True,
    "n_agents": n_agents,
    "cruise_altitude": 100,
    "add_value": 0.01,
    "seed": None,
    "save_risk_maps": saveRiskMap,
    "random_CD": False,
    "include_wind": False,
    "water_setting": 0,
    "correction_matrix": x
    # "modify_dens_scenario": 1
    # "water_setting": 0
    # "modify_dens_list": [0, 0.1693, 0, 0.0716, 0, 0.5488, 0, 0] # 0 = remove pop dens from water.
    # "modify_dens_list": [0, 0., 0, 0.0, 0, 0.965, 0, 0]
    # "modify_dens_scenario": "extreme_day"
}
# variable_params = {"modify_dens_scenario": [1, 2, 3, 4, 5, 6, 7, 8, 9], "city": ["Delft", "Paris", "NewYork"]}
# variable_params = {"modify_dens_scenario": [1, 2, 3, 4, 5, 6, 7, 8, 9]}

variable_params = {}
model_reporter_parameters = {"Flown distance": compute_flown_dis,
                             "Planned distance": compute_planned_dis,
                             "Optimal distance": compute_optimal_dis,
                             "Individual_risk_map": compute_individual_risk_map,
                             # "Collective risk": compute_collective_risk,
                             "Total Demand": compute_total_demand,
                             "Total Deliveries": compute_total_deliveries,
                             "Total Pickups": compute_total_pickups,
                             "position_heatmap": position_heatmap,
                             "crash_in_quantile": crash_in_quantile,
                             # "Heading list": return_heading_list,
                             "compute_flown_time_incl_unfinished": compute_flown_time_incl_unfinished,
                             "Time Waiting": compute_time_waiting,
                             "Time hovering (CDR)": compute_hover_time,
                             "Flighttime per shelter category": compute_time_in_shelter_category,
                             "Cruise Time": compute_time_cruise,
                             "Time Climb and Descent": compute_time_climb_descent,
                             "Takeoff and landing time": compute_time_takeoff_landing,
                             "density_matrix_scaled_risk": density_matrix_scaled_risk,
                             "area_risk_grid_m": area_risk_grid_m,
                             "Collective risk": compute_collective_risk,
                             "Collective risk cruise": compute_collective_risk_cruise,
                             "Collective risk c/d": compute_collective_risk_climb_descend,
                             "below_1": below_1,
                            "below_2": below_2,
                            "below_3": below_3,
                             "below_4": below_4,
                             "below_5": below_5,
                             "below_6": below_6,

                             "NNI_below_1": NNI_below_1,
                             "NNI_below_2": NNI_below_2,
                             "NNI_below_3": NNI_below_3,
                             "NNI_below_4": NNI_below_4,
                             "NNI_below_5": NNI_below_5,
                             "NNI_below_6": NNI_below_6,
                             "avg_risk_env": env_avg_risk,
                             "env_avg_risk_hubs": env_avg_risk_hubs,
                             "env_avg_risk_delivery_points": env_avg_risk_delivery_points,
                             "same_grid": same_grid,
                             "other_grid": other_grid,
                             "hub-location": hubpos,
                             "modify_alpha": modify_alpha,
                             "modify_beta": modify_beta,
                             "modify_shelter": modify_shelter,
                             "modify_mass": modify_mass
                             # "Crash Distance": compute_avg_distance_to_impact,
                             # "Crash Impact": compute_avg_E_on_impact
                             }
