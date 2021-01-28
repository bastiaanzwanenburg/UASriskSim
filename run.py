# -*- coding: utf-8 -*-
"""
Runs the model including a visualization. Best to use this file if debugging or observing emergent behavior. Performance of visualization is poor, so do not run this file if looking to gather a lot of results.
"""
import math

import numpy as np
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
# sys.path.extend(['/Users/bastiaan/PycharmProjects/AABMS/DroneDelivery_v0.2_cheng0305'])
from mesa.visualization.modules import CanvasGrid, ChartModule
from env.loadEnvironment import loadEnvironment

from server import agent_portrayal
from model import DeliveryModel


from parameters import n_agents, n_iterations, max_steps, model_reporter_parameters, fixed_params

"""The objective of run.py is to run a visualization of the model. """
# setups

# density_matrix = cityEnvironment.density_matrix
# shelter_map = cityEnvironment.shelter_matrix
#
# scaling_factor = 2
# Risk_map_resolution = 2
#
# density_matrix_scaled = cityEnvironment.scaleMap(density_matrix, scaling_factor=scaling_factor,
#                                                      scaling_method="original_AABMS")
# shelter_map_scaled = cityEnvironment.scaleMap(shelter_map, scaling_factor=scaling_factor,
#                                                   scaling_method="mode")
#
# density_matrix_scaled_risk = cityEnvironment.scaleMap(density_matrix, scaling_factor=Risk_map_resolution,
#                                                           scaling_method="original_AABMS")
# shelter_map_scaled_risk = cityEnvironment.scaleMap(shelter_map, scaling_factor=Risk_map_resolution,
#                                                        scaling_method="mode")
#
# shelter_category = cityEnvironment.shelter_category
# shelter_category_scaled = cityEnvironment.scaleMap(shelter_category, scaling_factor=scaling_factor,
#                                                    scaling_method="mode")  # TODO scale this appropriately for both risk and pathfinding
#
# shelter_category_scaled_risk = cityEnvironment.scaleMap(shelter_category, scaling_factor=scaling_factor,
#                                                         scaling_method="mode")
# shelter_category_names = cityEnvironment.shelter_category_names
#
# shelter_category_names = np.append(shelter_category_names, "hub")
# shelter_category_names = np.append(shelter_category_names, "delivery point")
#
# shelter_category = cityEnvironment.scaleMap(shelter_category, scaling_factor=scaling_factor, scaling_method="mode")
#
#
# width = density_matrix_scaled.shape[0]
# height = density_matrix_scaled.shape[1]
# width_risk = math.ceil(density_matrix.shape[0]/Risk_map_resolution)# Width of girds
# height_risk = math.ceil(density_matrix.shape[1]/Risk_map_resolution) # Height of girds

grid = CanvasGrid(agent_portrayal, 189, 187, 1134, 1122)
'''Creates a grid for visualization. '''

chart = ChartModule([{"Label": "Flown distance", "Color": "Black"},
                     {"Label": "Planned distance", "Color": "Red"},
                     {"Label": "Optimal distance", "Color": "Green"}],
                    data_collector_name='datacollector')

chart2 = ChartModule([{"Label": "Total travel time", "Color": "Black"},
                      {"Label": "Time hovering (CDR)", "Color": "Red"}],
                     data_collector_name='datacollector')

chart3 = ChartModule([{"Label": "Collective risk", "Color": "Black"}],
                     data_collector_name='datacollector')

chart4 = ChartModule([{"Label": "Total Deliveries", "Color": "Black"},
                      {"Label": "Total Orders", "Color": "Red"}],
                     data_collector_name='datacollector')

# if (density_matrix_scaled.shape != shelter_map_scaled.shape) or (density_matrix.shape != shelter_map.shape) or (
#         density_matrix_scaled_risk.shape != shelter_map_scaled_risk.shape):
#     raise Exception("Dimensions of density matrix and sheltering do not match")

server = ModularServer(DeliveryModel,
                       [grid, chart, chart2, chart3, chart4],
                       "Delivery Model",
                       fixed_params)
'''Starts server, which is run on localhost and presents visualization of model.'''


server.port = np.random.randint(8500, 9000)
'''Randomly select server port. If this is kept fixed, this gives issues when re-running because the port used previously is blocked.'''
# server.launch()

#
#

