'''
All metrics are stored using this file.
'''



from agents.agents import HubAgent, DeliveryPointAgent, ObstacleAgent, PopulationDensity
from agents.drone_agent import DroneAgent
from env.loadEnvironment import loadEnvironment
import numpy as np

'''File that includes all metrics of the model.'''

def compute_time_cruise(model):
    return model.time_cruise

def compute_time_climb_descent(model):
    return model.time_climb_descent

def compute_time_waiting(model):
    return model.time_waiting

def compute_time_takeoff_landing(model):
    return model.time_takeoff_landing

def compute_time_in_shelter_category(model):
    return model.flight_time_shelter_category

def crash_in_quantile(model):
    return [model.crash_in_quantile]

def hubpos(model):
    return [model.hubloc]

def below_1(model):
    return model.v_below_1
def below_4(model):
    return model.v_below_4

def below_5(model):
    return model.v_below_5

def below_6(model):
    return model.v_below_6

def same_grid(model):
    return model.same_grid

def other_grid(model):
    return model.other_grid

def below_2(model):
    return model.v_below_2

def below_3(model):
    return model.v_below_3

def env_avg_risk(model):
    return model.env_avg_risk

def env_avg_risk_hubs(model):
    return np.mean(model.env_values_around_hub)

def env_avg_risk_delivery_points(model):
    return np.mean(model.env_values_around_delivery_points)

def NNI_below_1(model):
    return model.NNI_below_1

def NNI_below_2(model):
    return model.NNI_below_2

def NNI_below_3(model):
    return model.NNI_below_3

def NNI_below_4(model):
    return model.NNI_below_4


def NNI_below_5(model):
    return model.NNI_below_5

def NNI_below_6(model):
    return model.NNI_below_6


def compute_total_deliveries(model):
    return model.total_deliveries

def compute_total_pickups(model):
    return model.total_pickups

def compute_total_demand(model):
    return model.total_demand

def compute_collective_risk(model):
    return model.collective_risk

def compute_collective_risk_climb_descend(model):
    return model.collective_risk_climb_descend

def compute_collective_risk_cruise(model):
    return model.collective_risk_cruise

def compute_avg_distance_to_impact(model):
    return [model.crash_dist]

def compute_avg_E_on_impact(model):
    return [model.Eimp]


def area_risk_grid_m(model):
    return model.area_risk_grid_m
# def return_heading_list(model):
#     return model.heading_list
def density_matrix_scaled_risk(model):
    return [model.shelter_map_scaled]

def compute_total_flown_time(model):
    # Only includes finished routes
    drone_agents = []
    for agent in model.schedule.agents:
        if type(agent) is DroneAgent:
            drone_agents.append(agent)
    total_flown_time = [a.total_flown_time for a in drone_agents]
    return sum(total_flown_time)

def compute_flown_time_incl_unfinished(model):
    # Only includes finished routes
    drone_agents = []
    for agent in model.schedule.agents:
        if type(agent) is DroneAgent:
            drone_agents.append(agent)
    flown_time_incl_unfinished = [a.total_flown_time_incl_unfinished for a in drone_agents]
    return sum(flown_time_incl_unfinished)


def modify_alpha(model):
    return model.modify_alpha

def modify_beta(model):
    return model.modify_beta

def modify_shelter(model):
    return model.modify_shelter

def modify_mass(model):
    return model.modify_mass



def compute_hover_time(model):
    drone_agents = []
    for agent in model.schedule.agents:
        if type(agent) is DroneAgent:
            drone_agents.append(agent)
    hover_time = [a.hover_time for a in drone_agents]
    return sum(hover_time)

def compute_planned_dis(model):
    drone_agents = []
    for agent in model.schedule.agents:
        if type(agent) is DroneAgent:
            drone_agents.append(agent)
    planned_dis = [a.planned_dis_total_completed for a in drone_agents]
    return sum(planned_dis)

def compute_optimal_dis(model):
    drone_agents = []
    for agent in model.schedule.agents:
        if type(agent) is DroneAgent:
            drone_agents.append(agent)
    optimal_dis = [a.optimal_dis_total_completed for a in drone_agents]
    return sum(optimal_dis)

def compute_flown_dis(model):
    drone_agents = []
    for agent in model.schedule.agents:
        if type(agent) is DroneAgent:
            drone_agents.append(agent)
    flown_dis = [a.flown_dis_total_completed for a in drone_agents]
    return sum(flown_dis)

def compute_individual_risk_map(model):
    return [model.risk_map_individual]

def position_heatmap(model):
    return [model.position_heatmap]

# def compute_collective_risk(model):
#     raise Exception("This reporter shouldn't be used, it is not efficient to do this at every iteration.")

    # NOTE: THE FOLLOWING CODE IS NOT CORRECT!!!!
    # Population_map = cityEnvironment.density_matrix[0:cityEnvironment.density_matrix.shape[0]:model.risk_map_resolution,0:cityEnvironment.density_matrix.shape[1]:model.risk_map_resolution]
    # collective_risk = np.multiply(model.risk_map_individual, Population_map)
    # return sum(sum(collective_risk))*10000*(3.8*4.6)/(749*909)# *10000 because we set population 10000 times lower in loadEnvironment.py
    # return sum(sum(collective_risk))*10000*(3.8*4.6)/(model.width_risk*model.height_risk)
