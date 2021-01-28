# -*- coding: utf-8 -*-
"""
main file that generates and simulates the ABM. Run this file directly if you only want to simulate the model without any visualization. Multi-processing is supported.
"""
import pickle
import random
import glob
from datetime import datetime

  # Used for benchmarking purposes
import resource
import numpy as np

from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner, BatchRunnerMP
import matplotlib.pyplot as plt
from env.loadEnvironment import loadEnvironment
from sklearn.neighbors import NearestNeighbors

from parameters import MP, n_agents, n_iterations, max_steps, model_reporter_parameters, fixed_params, variable_params

import math
# import random
from helpers.databases import routeDatabase, riskDatabase, crashLocationDatabase


from helpers.metrics import *

def NNI(data, width, height):
    if len(data) < 2:
        return -1

    expected_data = []
    for i in range(0, len(data)):
        expected_data.append((random.uniform(0, width), random.uniform(0, height)))

    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(data)
    distances, indices = nbrs.kneighbors(data)
    nearest_neighbor_distance = np.mean(distances[:, 1])

    # nbrs2 = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(expected_data)
    # distances_2, indices = nbrs2.kneighbors(expected_data)
    # exp_distance = np.mean(distances_2[:, 1])
    exp_distance = 0.5 / (len(data)/(width*height))**0.5 # same as above --> http://resources.esri.com/help/9.3/arcgisdesktop/com/gp_toolref/spatial_statistics_tools/how_average_nearest_neighbor_distance_spatial_statistics_works.htm
    # if (nearest_neighbor_distance / exp_distance) > 1:
    #     plt.scatter(*zip(*expected_data))
    #     plt.show()
    #     plt.scatter(*zip(*data))
    #     plt.show()
    #     print("break")
    return (nearest_neighbor_distance / exp_distance)



class DeliveryModel(Model):
    """A model with drone agents that pick up packages at a hub and delivery these at a delivery point.
    The switches allows the user to use no-fly zones, conflict detection and resolution and to choose the path finding algorithm.
    When choosing shelter_and_pop_density or pop_density, a RiskA* algorithm is used based on sheltering factors and population density,
    or solely on the population density. When choosing False, the model uses the regular A* algorithm."""
    def __init__(self,
                 n_agents,
                 city,
                 scaling_factor,
                 risk_map_resolution,
                 path_lockfile,
                 path_database,
                 FailureRateCruise,
                 FailureRateClimbDescend,
                 Risk_Optimal,
                 max_steps,
                 add_value,
                 n_random_hubs=0,
                 Hub_location=[],
                 n_random_delivery_locations=0,
                 Delivery_location=[],
                 random_hub_location="edges",
                 total_demand_per_hour=1,
                 scale_demand_with_density = False,
                 Nofly=False,
                 conflictdetection=False,
                 risk_path_planning_intervals=1,
                 add_noise_to_dens_list = False,
                 correction_matrix = [],
                 correction_scenario = 0,
                 mass_correction_factor = 1.0,
                 modify_alpha = 1.0,
                 shelter_correction_factor = 1.0,
                 modify_beta = 1.0,
                 cruise_altitude = 100,
                 modify_dens_scenario = 1,
                 random_CD = False,
                 include_wind = False,
                 save_risk_maps = False,
                 alpha=32000,
                 beta=34,
                 air_density=1.225,
                 gravitational_constant=9.81,
                 v_impact_scenario=1,
                 water_setting = 0,
                 seed=None):
        '''
        Args:
            n_agents: Number of drone agents
            city: Which city to model. Current options: Delft, NewYork, Paris.
            scaling_factor: Factor with which the risk-map is scaled.

            path_lockfile: Used for the database in which paths are stored in MP. Bit of a hacky work-around, used to make sure the databasefile is not overwritten by multiple processes at the same time, thus losing data.
            path_database: Database in which the actual paths are stored. Note: this only works well if DPs and hubs are placed at same place in each simulation, otherwise, there will be virtually no database hits.

            FailureRateCruise: P(failure) per second in cruise phase
            FailureRateClimbDescend: P(failure) per second in C&D phase
            Risk_Optimal: Setting how the path-finder works. Options: calculate_risk (i.e. fully calculate the risk by simulating a crash), shelter_and_pop_density (base risk on risk over a location), pop_density (base risk only on pop density at a location), False

            max_steps: Maximum number of steps to simulate.
            add_value: Heuristic value for path-finding.

            The model supports correction factors for parameters of the risk computation. These can either be supplied individually, or as a matrix:
            Option A: correction_matrix (has to have length 4)
            Option B: Specify following four parameters: mass_correction_factor, modify_alpha, shelter_correction_factor, modify_beta

            cruise_altitude: cruise altitude in meters
            modify_dens_scenario: can be one of 9 scenarios (so accepts parameters 1 - 9)

            random_CD: Boolean, whether or not to simulate a stochastic Cd
            include_wind: Boolean, whether or not to simulate random wind
            save_risk_maps: Whether or not to store the actual risk maps. Increases computational effort significantly but can be used for reporting / analysis.

            alpha: Value of the fatality model
            beta: Value of the fatality model

            air_density: in kg/m^3
            gravitational_constant: gravitational acceleration in m/s^2
            v_impact_scenario: 1 = modeled impact speed, 2 = terminal velocity, 3 = terminal velocity + 40%
            water_setting: if 0, remove all population from water. Else, do nothing.

            seed: can be used to seed all RNGs in the model.


        '''
        super().__init__()

        self.startTime = datetime.now()
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        else:
            self.seed = None

        if len(correction_matrix) == 4:
            self.modify_mass = correction_matrix[correction_scenario, 0]
            self.modify_alpha = correction_matrix[correction_scenario, 1]
            self.modify_beta = correction_matrix[correction_scenario, 2]
            self.modify_shelter = correction_matrix[correction_scenario, 3]
        else:
            self.modify_mass = mass_correction_factor
            self.modify_alpha = modify_alpha
            self.modify_beta = modify_beta
            self.modify_shelter = shelter_correction_factor



        self.alpha = alpha * self.modify_alpha # impact energy required for a fatality probability of 50% when ps = 6
        self.beta = beta * self.modify_alpha # impact energy needed to cause a fatality when ps approaches zero

        self.shelter_correction_factor = self.modify_shelter
        self.mass_correction_factor = self.modify_mass

        # Usually: ['Error' 'Office' 'Water' 'Street' 'Event Location' 'Home Area', 'Small Street' 'Meadow', Hub, DP]
        if modify_dens_scenario == 1:
            self.modify_dens_list = [0, 0.377, 0, 0.052, 0, 0.559, 0.0, 0.052, 0, 0]

        elif modify_dens_scenario == 2:
            self.modify_dens_list = [0, 0.0389, 0, 0, 0, 0.958, 0, 0.0031, 0, 0]

        elif modify_dens_scenario == 3:
            self.modify_dens_list = [0, 0.077, 0.039, 0.01, 0, 0.626, 0.169, 0.088, 0, 0]

        elif modify_dens_scenario == 4:
            self.modify_dens_list = [0, 0.27, 0.08, 0.091, 0, 0.509, 0, 0.049, 0, 0]

        elif modify_dens_scenario == 5:
            self.modify_dens_list = [0, 0.19365, 0.0195, 0.0505, 0, 0.5874, 0.0845, 0.0684, 0, 0]

        elif modify_dens_scenario == 6:
            self.modify_dens_list = [0, 0.05795, 0.0195, 0.005, 0, 0.792, 0.0845, 0.04555, 0, 0]

        elif modify_dens_scenario == 7:
            self.modify_dens_list = [0, 0.1735, 0.0595, 0.0505, 0, 0.5675, 0.0845, 0.0685, 0, 0]

        elif modify_dens_scenario == 8:
            self.modify_dens_list = [0, 0.29015, 0.04, 0.091, 0, 0.5289, 0, 0.0489, 0, 0]

        elif modify_dens_scenario == 9:
            self.modify_dens_list = [0, 0.15445, 0.04, 0.0455, 0, 0.7335, 0, 0.02605, 0, 0]

        else:
            self.modify_dens_list = []


        if add_noise_to_dens_list:
            # Adds noise to the density lists.

            self.modify_dens_list = [np.random.normal(x, 0.015) for x in self.modify_dens_list]
            self.modify_dens_list = [0 if x < 0 else x for x in self.modify_dens_list]
            self.modify_dens_list = [1 if x > 1 else x for x in self.modify_dens_list]

       #  if modify_dens_scenario == "day":
       #      self.modify_dens_list = [0, 0.1693, 0, 0.0716, 0, 0.5488, 0, 0]
       #  elif modify_dens_scenario == "night":
       #      self.modify_dens_list = [0, 0.035, 0, 0.0, 0, 0.965, 0, 0]
       #  elif modify_dens_scenario == "extreme_day":
       #      self.modify_dens_list = [0, 0.4, 0, 0.1, 0, 0.5, 0, 0]
       #  elif modify_dens_scenario == "event":
       #      self.modify_dens_list = [0, 0.08, 0, 0.07, 0.3544, 0.2774, 0.,0.]
       #  else:
       #      self.modify_dens_list = []


        self.modify_dens_scenario = modify_dens_scenario
        self.same_grid = 0 # Counter how many crashes are on the same grid as the drone's position durin
        self.other_grid = 0 # Idem, but then on other grid.

        self.hubloc = []

        self.scale_demand_with_density = scale_demand_with_density
        self.include_wind = include_wind
        self.random_CD = random_CD
        self.save_risk_maps = save_risk_maps

        # Statistics on CR
        self.collective_risk = 0
        self.collective_risk_cruise = 0
        self.collective_risk_climb_descend = 0

        self.vimp = [] # A list in which all v_imps of all crashes are stored, used for statistics


        self.add_value = add_value

        # Parameters of locations in the environment
        self.hub_location = Hub_location
        self.n_random_hubs = n_random_hubs
        self.random_hub_location = random_hub_location
        self.delivery_location = Delivery_location
        self.n_random_delivery_locations = n_random_delivery_locations

        # Some checks to see if hubs and delivery points are correct
        if self.n_random_hubs == 0 and len(self.hub_location) == 0:
            raise Exception("No hub location specified")
        elif self.n_random_hubs > 4:
            raise Exception("More than four edges to place hubs on")
        elif self.n_random_hubs > 0 and len(self.hub_location) > 0:
            raise Exception("We cannot combine random hubs and placed hubs")

        if self.n_random_delivery_locations == 0 and len(self.delivery_location) == 0:
            raise Exception("No delivery location specified")
        elif self.n_random_delivery_locations > 0 and len(self.delivery_location) > 0:
            raise Exception("We cannot combine random delivery locations and placed delivery locations")

        # Variables for risk computation


        self.impact_list = []
        self.impact_angle_list = []
        self.air_density = air_density
        self.gravitational_constant = gravitational_constant

        self.crash_in_quantile = np.zeros(4) # (0 = total, 1 = <0.1, 2 = <0.05, 3 = <0.01)

        self.obstacle_list = []
        self.nofly_list = []
        self.no_fly = Nofly

        self.crash_dist = []
        self.Eimp = []

        # print("Started a new iteration with S:{}, R:{}".format(scaling_factor, risk_map_resolution))
        self.risk_path_planning_intervals = risk_path_planning_intervals

        self.v_impact_scenario = v_impact_scenario # 1 = based on modeled impact speed, 2 = terminal velocity, 3 = 40% above terminal velocity.

        self.cruise_altitude = cruise_altitude

        # For demand
        self.demand_per_hour = total_demand_per_hour
        self.max_steps = max_steps

        # Environment maps

        self.scaling_factor = scaling_factor

        cityEnvironment = loadEnvironment(city=city, scaling_factor=scaling_factor,
                                          risk_map_resolution=risk_map_resolution, water_setting = water_setting, modify_dens_list = self.modify_dens_list)

        self.density_matrix = cityEnvironment.density_matrix.copy()
        self.shelter_map = cityEnvironment.shelter_matrix.copy()  # shelter_matrix factor map

        self.shelter_map = np.multiply(self.shelter_map, self.shelter_correction_factor)

        self.density_matrix_scaled = cityEnvironment.density_matrix_scaled.copy()
        self.shelter_map_scaled = cityEnvironment.shelter_map_scaled.copy()

        self.shelter_map_scaled = np.multiply(self.shelter_map_scaled, self.shelter_correction_factor)

        self.density_matrix_scaled_risk = cityEnvironment.density_matrix_scaled_risk.copy()

        self.shelter_map_scaled_risk = cityEnvironment.shelter_map_scaled_risk.copy()

        self.shelter_map_scaled_risk = np.multiply(self.shelter_map_scaled_risk, self.shelter_correction_factor)

        self.shelter_category = cityEnvironment.shelter_category.copy()
        self.shelter_category_scaled = cityEnvironment.shelter_category_scaled.copy()
        self.shelter_category_scaled_risk = cityEnvironment.shelter_category_scaled_risk.copy()
        self.shelter_category_names = cityEnvironment.shelter_category_names.copy()
        self.area_risk_grid_m = cityEnvironment.area_risk_m

        self.density_matrix_scaled_risk_area_grid_risk_m = self.density_matrix_scaled_risk * self.area_risk_grid_m

        self.flight_time_shelter_category = np.zeros((len(self.shelter_category_names)))

        # Environment dimensions
        self.width = self.density_matrix_scaled.shape[0]
        self.height = self.density_matrix_scaled.shape[1]
        # print("Width = {}, height = {}".format(self.width, self.height))
        self.risk_map_resolution = risk_map_resolution
        self.width_risk = self.density_matrix_scaled_risk.shape[0]
        self.height_risk = self.density_matrix_scaled_risk.shape[1]

        # print("Model dimensions = {} x {}".format(self.width_risk, self.height_risk))

        self.risk_grid_size_m_x = cityEnvironment.risk_grid_size_m_x
        self.risk_grid_size_m_y = cityEnvironment.risk_grid_size_m_y


        # print(self.area_risk_grid_m)
        if (self.risk_grid_size_m_x * self.risk_grid_size_m_y) != self.area_risk_grid_m:
            raise Exception("Error!")

        self.environment_real_area = cityEnvironment.width_meters * cityEnvironment.height_meters

        self.grid_size_m_x = cityEnvironment.grid_size_m_x
        self.grid_size_m_y = cityEnvironment.grid_size_m_y

        self.routes = dict()

        self.total_risk = np.zeros((self.density_matrix_scaled_risk.shape[0], self.density_matrix_scaled_risk.shape[1]))

        # Size of one vertex of a grid (in meters) in the x-direction
        # Size of one vertex of aa grid (in meters) in the y-direction

        # Agents parameters
        self.num_agents = n_agents
        self.total_deliveries = 0
        self.total_pickups = 0
        self.total_demand = 0
        self.failure_rate_cruise = FailureRateCruise
        self.failure_rate_climb_descend = FailureRateClimbDescend
        self.conflictdetection = conflictdetection

        self.time_cruise = 0
        self.time_takeoff_landing = 0
        self.time_climb_descent = 0
        self.time_waiting = 0

        self.risk_hit = 0
        self.risk_miss = 0

        # Simulation setup parameters
        self.grid = MultiGrid(self.width, self.height, False)
        self.schedule = RandomActivation(self)
        self.running = True
        self.angles = np.zeros(4)

        self.riskDatabase = riskDatabase()

        self.crashLocationDatabase = crashLocationDatabase()

        self.path_lockfile = path_lockfile
        self.path_database = path_database
        with open(self.path_lockfile, "w+") as myfile:
            myfile.write('t')


        with open(self.path_database, "wb+") as dbfile:
            pickle.dump(dict(), dbfile)

        self.routeDatabase = routeDatabase(loadDatabaseFromFile=False, db_url=self.path_database,
                                           lock_url=self.path_lockfile)

        # Risk calculation parameters
        self.risk_optimal = Risk_Optimal

        self.min_risk_shelter_and_pop = 1e6
        self.max_risk_shelter_and_pop = 0
        # print("Width x height = {} x {}".format(self.grid.width, self.grid.height))
        for i in range(self.grid.width):
            for j in range(self.grid.height):

                # Impact kinetic energy
                v_terminal = 50
                Eimp = 0.5 * 3 * v_terminal ** 2


                # Falality probability
                ps = self.shelter_map_scaled[i, j]
                k = min(1, pow((self.beta / Eimp), 3 / ps))
                value = pow((self.alpha / self.beta), 0.5) * pow((self.beta / Eimp), 3 / (ps))
                pf = (1 - k) / (1 - 2 * k + value)
                value = pf * self.density_matrix_scaled[i, j]

                if value < self.min_risk_shelter_and_pop:
                    self.min_risk_shelter_and_pop = value
                if value > self.max_risk_shelter_and_pop:
                    self.max_risk_shelter_and_pop = value



        # self.min_risk_shelter_and_pop = self.min_risk_shelter_and_pop + self.add_value

        # Find RC_min [Used for RiskA*, see also Primatesta (2019)]
        if self.risk_optimal == "pop_density":
            self.min_risk = self.density_matrix_scaled.min()

        elif self.risk_optimal == "calculate_risk":
            self.min_risk = 1 # TODO: Check if this value is right.

        elif self.risk_optimal == "shelter_and_pop_density":
            self.min_risk = self.min_risk_shelter_and_pop
            if self.min_risk == 1e6:
                raise Exception("Min risk still the original value")
            # else:
            #     print("Min risk = {}".format(self.min_risk))
        else:
            self.min_risk = 1

        self.min_risk += self.add_value

        self.risk_map_individual = np.zeros((self.width_risk, self.height_risk))
        self.position_heatmap = np.zeros((self.width, self.height))
        self.max_risk_iterations = int(1e3)  # Max iterations per risk calculation (to account for stochastic drag coeff C_d)

        self.datacollector = DataCollector(model_reporter_parameters)

        # define population density
        if __name__ != "__main__":
            for i in range(self.grid.width):
                for j in range(self.grid.height):
                    density = PopulationDensity((i * 100 + j), self)
                    x = i
                    y = j
                    self.grid.place_agent(density, (x, y))
                    # density.colorchange()

                    ps = self.shelter_map_scaled[i, j]
                    # Impact kinetic energy
                    v_terminal = 50
                    Eimp = 0.5 * 3 * v_terminal ** 2
                    # alpha = 32000  # impact energy required for a fatality probability of 50% when ps = 6
                    # beta = 34  # impact energy needed to cause a fatality when ps approaches zero

                    # Falality probability
                    k = min(1, pow((beta / Eimp), 3 / ps))
                    value = pow((alpha / beta), 0.5) * pow((beta / Eimp), 3 / (ps))
                    pf = (1 - k) / (1 - 2 * k + value)
                    value = pf * self.density_matrix_scaled[i, j]

                    density.density = value / self.max_risk_shelter_and_pop
                    # if value > 1 or value < 0:
                    #     print("wtf")
                    # if statement for visualization of no-fly zones
                    if (x, y) in self.obstacle_list:
                        density.density = 10

        # create Hub(s) at given position.
        self.hub_pos = []

        n_fixed_hub_locations = len(self.hub_location)

        n_hubs = max(n_fixed_hub_locations, self.n_random_hubs) # One of the two is 0 (this is checked in the __init__

        self.demand_per_hub_per_hour = math.floor(self.demand_per_hour / n_hubs)

        # Some demand won't be allocated. Note this just in case.
        self.demand_not_allocated = self.demand_per_hour - n_hubs * self.demand_per_hub_per_hour
        if self.demand_not_allocated > 0:
            print("WARN: Unallocated demand per hour of {}".format(self.demand_not_allocated))
        # Two scenarios:
        # 1) n_fixed_hub_locations = fixed locations
        # 2) self.n_random_hubs = random hub locations



        if n_fixed_hub_locations > 0:
            #This is the scenario where we have fixed hub locations
            for i in range(len(Hub_location)):
                hub = HubAgent(i + 50000, self, unlimited_demand=False, demand_per_hour=self.demand_per_hub_per_hour, demand_generation_interval="poisson")
                x = self.hub_location[i][0]
                y = self.hub_location[i][1]
                if not (x, y) in self.obstacle_list:
                    self.schedule.add(hub)
                    self.grid.place_agent(hub, (x, y))
                    self.hub_pos.append((x, y))

                else:
                    print("Note: Hub {} relocated because it would coincide with an obstacle".format(i))
                    positioning = 0
                    while positioning < 1:
                        x = self.random.randrange(self.grid.width)
                        y = self.random.randrange(self.grid.height)
                        if not (x, y) in self.obstacle_list:
                            self.schedule.add(hub)
                            self.grid.place_agent(hub, (x, y))
                            self.hub_pos.append((x, y))
                            positioning = 1

        elif self.n_random_hubs > 0:
            #This is the scenario where we are placing random hubs

            # Create a list with positions = all the edges of the map.
            possible_hub_locations = []

            if self.random_hub_location == "edges":
                possible_hub_locations.append((0, self.random.randint(0, self.grid.height - 1)))  # pos = (0,randh)
                possible_hub_locations.append((self.grid.width - 1, self.random.randint(0, self.grid.height - 1)))
                possible_hub_locations.append((self.random.randint(0, self.grid.width - 1), 0))
                possible_hub_locations.append((self.random.randint(0, self.grid.width - 1), self.grid.height - 1))


                for i in range(0, self.n_random_hubs):
                    hubloc = self.random.choice(possible_hub_locations)
                    possible_hub_locations.remove(hubloc)

                    hub = HubAgent(i + 50000, self, unlimited_demand=False, demand_per_hour=self.demand_per_hub_per_hour, demand_generation_interval="poisson")
                    x = hubloc[0]
                    y = hubloc[1]
                    if not (x, y) in self.obstacle_list:
                        self.schedule.add(hub)
                        self.grid.place_agent(hub, (x, y))
                        self.hub_pos.append((x, y))
                    else:
                        raise Exception("Random hub location placement not working")

                self.hubloc = self.hub_pos
            elif self.random_hub_location == "random":
                if self.n_random_hubs is not 1:
                    raise Exception("Not sure whether it is possible to place more than 1 random hubb")

                hubloc = (self.random.randint(0, self.grid.width - 1), self.random.randint(0, self.grid.height - 1))
                hub = HubAgent(i + 50000, self, unlimited_demand=False, demand_per_hour=self.demand_per_hub_per_hour,
                               demand_generation_interval="poisson")
                x = hubloc[0]
                y = hubloc[1]
                self.hubloc = hubloc
                if not (x, y) in self.obstacle_list:
                    self.schedule.add(hub)
                    self.grid.place_agent(hub, (x, y))
                    self.hub_pos.append((x, y))
                else:
                    raise Exception("Random hub location placement not working")
            else:
                raise Exception("Don't know where to place hubs.")

            for hubpos in self.hub_pos:
                # Do this before placing delivery locations such that hub-locations are not tried for delivery locations.
                # print("Hub pos {}".format(hubpos))
                self.shelter_category_scaled[hubpos] = int(np.where(self.shelter_category_names == "hub")[0][0])


        else:
            raise Exception("We can't place any hubs")


        # create Delivery point(s) at given position

        self.all_delivery_positions = []
        if len(self.delivery_location) > 0:
            for i in range(len(self.delivery_location)):
                deliverypoint = DeliveryPointAgent(i, self)
                x = self.delivery_location[i][0]
                y = self.delivery_location[i][1]
                if (not (x, y) in self.obstacle_list) and (not (x, y) in self.hub_pos):
                    self.schedule.add(deliverypoint)
                    self.grid.place_agent(deliverypoint, (x, y))
                    self.all_delivery_positions.append((x, y))
                else:
                    print("Note: Delivery point {} relocated because it would coincide with an obstacle or a hub".format(i))
                    positioning = 0
                    while positioning < 1:
                        x = self.random.randrange(self.grid.width)
                        y = self.random.randrange(self.grid.height)
                        if (not (x, y) in self.obstacle_list) and (not (x, y) in self.hub_pos):
                            self.grid.place_agent(deliverypoint, (x, y))
                            self.all_delivery_positions.append((x, y))
                            positioning = 1

        elif self.n_random_delivery_locations > 0:
            # Generate all possible delivery locations
            # And generate a list with the population density at that delivery location!
            self.possible_delivery_locations_population_density = []
            self.possible_delivery_locations = []
            for i in range(0, self.grid.width):
                for j in range(0, self.grid.height):
                    delivery_pos = (i, j)
                    pos_category = self.shelter_category_scaled[delivery_pos]
                    deliver_pos_cat_name = self.shelter_category_names[pos_category]
                    if deliver_pos_cat_name == "Office" or deliver_pos_cat_name == "Home Area":
                        self.possible_delivery_locations.append(delivery_pos)
                        self.possible_delivery_locations_population_density.append(self.density_matrix_scaled[delivery_pos])
            if len(self.possible_delivery_locations) < self.n_random_delivery_locations:
                raise Exception("Not enough places to place delivery locations")
            else:
                max_placement_tries = 100
                for i in range(0, self.n_random_delivery_locations):
                    placement_tries = 0
                    deliverypoint = DeliveryPointAgent(i, self)

                    while placement_tries < max_placement_tries:
                        # Weights = list with population density corresponding to the delivery location.
                        if self.scale_demand_with_density:
                            delivery_pos = self.random.choices(self.possible_delivery_locations, weights=self.possible_delivery_locations_population_density)[0]
                        else:
                            delivery_pos = self.random.choices(self.possible_delivery_locations)[0]


                        x = delivery_pos[0]
                        y = delivery_pos[1]

                        if (not (x, y) in self.obstacle_list) and (not (x, y) in self.hub_pos) and (not (x,y) in self.all_delivery_positions):
                            self.schedule.add(deliverypoint)
                            self.grid.place_agent(deliverypoint, (x, y))
                            self.all_delivery_positions.append((x, y))
                            break
                        else:
                            placement_tries += 1
        else:
            raise Exception("can't place delivery locations")

        for deliverpos in self.all_delivery_positions:
            self.shelter_category_scaled[deliverpos] = int(
                np.where(self.shelter_category_names == "delivery point")[0][0])

        self.gen_env_metrics()

        # # plt.imshow(self.shelter_map_scaled.transpose())
        # for deliverpos in self.all_delivery_positions:
        #     plt.scatter(deliverpos[0], deliverpos[1], color='r')
        # for hubpos in self.hub_pos:
        #     plt.scatter(hubpos[0], hubpos[1], color='m')
        #
        # plt.show()
        # Create drone agents
        for i in range(self.num_agents):
            a = DroneAgent(i, self, self.random.choice(self.hub_pos), (self.all_delivery_positions))
            self.schedule.add(a)
            self.grid.place_agent(a, a.hub_pos)

    '''Model steps '''

    def gen_env_metrics(self):
        values = []
        value_table = np.zeros_like(self.density_matrix_scaled)
        self.below_1_positions = []
        self.below_2_positions = []
        self.below_3_positions = []
        self.below_4_positions = []
        self.below_5_positions = []
        self.below_6_positions = []
        self.below_1_values = []
        self.below_2_values = []
        self.below_3_values = []
        self.below_4_values = []
        self.below_5_values = []
        self.below_6_values = []
        for i in range(self.grid.width):
            for j in range(self.grid.height):

                ps = self.shelter_map_scaled[i, j]
                # Impact kinetic energy
                v_terminal = 50
                Eimp = 0.5 * 3 * v_terminal ** 2
                # alpha = 32000  # impact energy required for a fatality probability of 50% when ps = 6
                # beta = 34  # impact energy needed to cause a fatality when ps approaches zero

                # Falality probability
                k = min(1, pow((self.beta / Eimp), 3 / ps))
                value = pow((self.alpha / self.beta), 0.5) * pow((self.beta / Eimp), 3 / (ps))
                pf = (1 - k) / (1 - 2 * k + value)
                value = pf * self.density_matrix_scaled[i, j]  # density_matrix unit = people / m^2
                # The unit of value = the amount of people that would die, given an impact in this place
                value_table[i,j] = value
                values.append(value)

        # self.threshold_5 = 0.001
        self.threshold_1 = np.quantile(values, 0.10) #0.02
        self.threshold_2 = np.quantile(values, 0.20) #0.01
        self.threshold_3 = np.quantile(values, 0.25) #0.001

        self.threshold_4 = 0.001
        self.threshold_5 = self.threshold_4 * 0.75
        self.threshold_6 = self.threshold_4 * 1.25

        #
        # threshold_4 = np.quantile(values, 0.9)
        # threshold_5 = np.quantile(values, 0.95)
        # threshold_6 = np.quantile(values, 0.99)

        for i in range(self.grid.width):
            for j in range(self.grid.height):
                value = value_table[i, j]

                if value <= self.threshold_3:
                    self.below_3_positions.append((i,j))
                    self.below_3_values.append(value_table[i,j])
                if value <= self.threshold_2:
                    self.below_2_positions.append((i,j))
                    self.below_2_values.append(value_table[i,j])

                if value <= self.threshold_1:
                    self.below_1_positions.append((i,j))
                    self.below_1_values.append(value_table[i,j])

                if value <= self.threshold_4:
                    self.below_4_positions.append((i,j))
                    self.below_4_values.append(value_table[i,j])

                if value <= self.threshold_5:
                    self.below_5_positions.append((i,j))
                    self.below_5_values.append(value_table[i,j])

                if value <= self.threshold_6:
                     self.below_6_positions.append((i,j))
                     self.below_6_values.append(value_table[i,j])


        self.NNI_below_1 = NNI(self.below_1_positions, self.grid.width, self.grid.height)
        self.NNI_below_2 = NNI(self.below_2_positions, self.grid.width, self.grid.height)
        self.NNI_below_3 = NNI(self.below_3_positions, self.grid.width, self.grid.height)
        self.NNI_below_4 = NNI(self.below_4_positions, self.grid.width, self.grid.height)
        self.NNI_below_5 = NNI(self.below_5_positions, self.grid.width, self.grid.height)
        self.NNI_below_6 = NNI(self.below_6_positions, self.grid.width, self.grid.height)


        threshold = 0.001
        plt.figure()
        value_list = value_table.reshape((value_table.shape[0]*value_table.shape[1]))
        plt.hist(value_list, bins=100)
        print("In scenario {}, it is {}%".format(self.modify_dens_scenario, (value_list <= threshold).sum()))
        plt.xlim([-0.001, 0.025])
        plt.title("Histogram of all grids on the map (scenario:NY, {})".format(self.modify_dens_scenario))
        plt.vlines(self.threshold_1, 0, 7289)
        plt.vlines(self.threshold_2, 0, 6289)
        plt.vlines(self.threshold_3, 0, 5289)
        plt.text(self.threshold_1, 7150, r"$\leftarrow$ 10% <= {}".format(round(self.threshold_1,4)), rotation=0, fontsize=11)
        plt.text(self.threshold_2, 6150, r"$\leftarrow$ 20% <= {}".format(round(self.threshold_2,4)), rotation=0, fontsize=11)
        plt.text(self.threshold_3, 5150, r"$\leftarrow$ 25% <= {}".format(round(self.threshold_3,4)), rotation=0, fontsize=11)
        plt.xlabel("Exp. no. of fatalities per grid, upon a standard impact.")
        filename = "hist_NY_scenario_{}.png".format(self.modify_dens_scenario)
        plt.savefig(filename)

        plt.show()

        # img = plt.imread("env/city_inputs/paris/paris_dens_map_v3_scaled_5.png")
        # fig, ax = plt.subplots()
        # ax.imshow(img)
        # # self.below_3_positions = np.multiply(self.below_3_positions, 5)
        # ax.scatter(*zip(*(self.below_3_positions)), s=1.0, color="m")
        # plt.title("ANN = {}, scenario = {}".format(round(self.NNI_below_3, 2), self.modify_dens_scenario))
        # # plt.colorbar()
        # # plt.scatter(*zip(*self.below_1_positions), s=2, color="m")
        # # str_fig = "data_thesis/env_exp/figures/map,ANN=" + str(round(self.NNI_below_3,2)) + ".png"
        # # plt.savefig(str_fig)
        # plt.show()


        # img = plt.imread("env/city_inputs/newyork/ny_corrected_map.png")
        # fig, ax = plt.subplots()
        # ax.imshow(img)
        # ax.scatter(*zip(*self.below_3_positions), s=2, color="m")
        # plt.title("ANN = {}".format(round(self.NNI_below_3,3)))
        # # plt.colorbar()
        # # plt.scatter(*zip(*self.below_1_positions), s=2, color="m")
        # plt.show()
        self.env_avg_risk = np.mean(values)

        # Risk around hub
        self.env_values_around_hub = []
        self.env_values_around_delivery_points = []
        for hub_pos in self.hub_pos:
            i = hub_pos[0]
            j = hub_pos[1]
            ps = self.shelter_map_scaled[i, j]
            # Impact kinetic energy
            v_terminal = 50
            Eimp = 0.5 * 3 * v_terminal ** 2
            # alpha = 32000  # impact energy required for a fatality probability of 50% when ps = 6
            # beta = 34  # impact energy needed to cause a fatality when ps approaches zero

            # Falality probability
            k = min(1, pow((self.beta / Eimp), 3 / ps))
            value = pow((self.alpha / self.beta), 0.5) * pow((self.beta / Eimp), 3 / (ps))
            pf = (1 - k) / (1 - 2 * k + value)
            value = pf * self.density_matrix_scaled[i, j] # density_matrix unit = people / m^2
            self.env_values_around_hub.append(value)

        for delivery_pos in self.all_delivery_positions:
            i = delivery_pos[0]
            j = delivery_pos[1]
            ps = self.shelter_map_scaled[i, j]
            # Impact kinetic energy
            v_terminal = 50
            Eimp = 0.5 * 3 * v_terminal ** 2
            alpha = 32000  # impact energy required for a fatality probability of 50% when ps = 6
            beta = 34  # impact energy needed to cause a fatality when ps approaches zero

            # Falality probability
            k = min(1, pow((self.beta / Eimp), 3 / ps))
            value = pow((self.alpha / self.beta), 0.5) * pow((self.beta / Eimp), 3 / (ps))
            pf = (1 - k) / (1 - 2 * k + value)
            value = pf * self.density_matrix_scaled[i, j]  # density_matrix unit = people / m^2
            self.env_values_around_delivery_points.append(value)


        # Risk at delivery points

        # self.v_below_1 = len(below_1_values) / self.grid.width / self.grid.height
        # self.v_below_2 = len(below_2_values) / self.grid.width / self.grid.height
        # self.v_below_3 = len(below_3_values) / self.grid.width / self.grid.height
        # self.v_below_4 = len(below_4_values) / self.grid.width / self.grid.height
        # self.v_below_5 = len(below_5_values) / self.grid.width / self.grid.height
        # self.v_below_6 = len(below_6_values) / self.grid.width / self.grid.height

        self.v_below_1 = np.mean(self.below_1_values)
        self.v_below_2 = np.mean(self.below_2_values)
        self.v_below_3 = np.mean(self.below_3_values)
        self.v_below_4 = np.mean(self.below_4_values)
        self.v_below_5 = np.mean(self.below_5_values)
        self.v_below_6 = np.mean(self.below_6_values)




        # print("Showing plot")
        # plt.hist(values, bins=25)
        # plt.ylim([0, 15000])
        # plt.xlim([0, 0.01])
        # plt.show()


    def step(self):
        self.schedule.step()
        # if self.schedule.steps % 200 == 0:
        #     print('debug here')
        # if self.schedule.steps == 999:
        #     plt.hist(self.impact_list)
        #     plt.show()
        #     print("debug")



             # self.avg_impact = sum(self.Eimp) / len(self.Eimp)
             # self.avg_dist = sum(self.crash_dist) / len(self.crash_dist)

        # if self.schedule.steps == 43200-1:
        #     print("Completed S:{}, R:{} in {}".format(self.scaling_factor, self.risk_map_resolution, datetime.now() - self.startTime))
        if self.schedule.steps == 1:
            self.startTime = datetime.now()
        # if self.schedule.steps % 1000 == 0:
        #     print("At step {}, 1000 steps in {}".format(self.schedule.steps, datetime.now() - self.startTime))
        #     self.startTime = datetime.now()

        # if self.schedule.steps % 10 == 0:
        #     print("Length of crash loc database = {}".format(len(self.crashLocationDatabase.locations_database)))
        # It is not necessary to do data-collection in every step.
        # if self.schedule.steps > 49900:
        #     self.datacollector.collect(self)
        if __name__ != "__main__":
            self.datacollector.collect(self)
        # #
        # if self.schedule.steps % 1000 == 0:
        #     import sys
        #     print("At step {}".format(self.schedule.steps))

        # Detect whether drone crashes on same grid
        # if (self.schedule.steps+1) % 100 == 0 and self.other_grid > 0:
        #     len_t = self.same_grid + self.other_grid
        #     print("Step={}: {} ({}%) crashes on same grid, {} ({}%) crashes on other grid".format(self.schedule.steps, self.same_grid, 100*self.same_grid/len_t, self.other_grid, 100*self.other_grid/len_t))
        # self.datacollector.collect(self)



## Initialize maps

''' Batch run '''
# Parameters
if __name__ == "__main__":

    if MP:
        batch_run = BatchRunnerMP(DeliveryModel,
                                nr_processes=30,
                                fixed_parameters=fixed_params,
                                variable_parameters=variable_params,
                                iterations=n_iterations,
                                max_steps=max_steps,
                                model_reporters=model_reporter_parameters,
                                )
    elif not MP:
        batch_run = BatchRunner(DeliveryModel,
                                fixed_parameters=fixed_params,
                                variable_parameters=variable_params,
                                iterations=n_iterations,
                                max_steps=max_steps,
                                model_reporters=model_reporter_parameters,
                                )
    batch_run.run_all()

    # Plot data
    run_data = batch_run.get_model_vars_dataframe()
    # run_data.head()
    # run_data["Collective risk"] = run_data["Individual_risk_map"].apply(
    #     lambda x: sum(sum(np.multiply(x, density_matrix_scaled_risk))) * 10000 * (3.8 * 4.6) / (
    #             width_risk * height_risk))

    run_data["Collective risk"] = run_data.apply(
        lambda row: np.sum((np.multiply(row["Individual_risk_map"][0], row["area_risk_grid_m"] * row["density_matrix_scaled_risk"][0]))) if row["save_risk_maps"] == True else row["Collective risk"], axis=1)

    # run_data["Collective risk"] = run_data["Individual_risk_map"].apply(
    #     lambda x: sum(sum(np.multiply(x, density_matrix_scaled_risk))) *area_risk_ )

    # run_data["Collective risk"] = run_data["Individual_risk_map"].apply(
    #     lambda x: sum(sum(np.multiply(x[0],
    #                                   cityEnvironment.area_risk_m * cityEnvironment.density_matrix_scaled_risk))))

    # total_flighttime = run_data["Cruise Time"] +

    # if abs((run_data["Collective risk"].mean() - run_data["Collective risk cruise"].mean() - run_data["Collective risk c/d"].mean())) / run_data["Collective risk"].mean() > 0.0001:
    #     raise Exception("Risks do not add up")

    run_data["Collective risk"] = run_data["Collective risk"] / 1000 # / 1000 reflects a correction of P(failure by a factor 1000)
    run_data["Collective risk cruise"] = run_data["Collective risk cruise"] / 1000 # / 1000 reflects a correction of P(failure by a factor 1000)
    run_data["Collective risk c/d"] = run_data["Collective risk c/d"] / 1000 # / 1000 reflects a correction of P(failure by a factor 1000)
    run_data["Total travel time"] = run_data["Time Climb and Descent"] + run_data["Cruise Time"]
    run_data["fatalities_per_1000_flight_hours"] = run_data["Collective risk"] * 3600 * 1000 / run_data["Total travel time"]
    run_data["fatalities_per_1000_flight_hours_cruise"] = run_data["Collective risk cruise"] * 1000 * 3600 / run_data["Cruise Time"]
    run_data["fatalities_per_1000_flight_hours_climb_descend"] = run_data["Collective risk c/d"] * 1000 * 3600 / run_data["Time Climb and Descent"]


    # plt.scatter(run_data.n_agents, run_data.flown_dis)

    # print("Risk = " + str(
    #     round(pow(10, 4) * run_data["Collective risk"].mean(), 3)) + ", Optimal vs flown distance = " + str(
    #     round(run_data["Optimal distance"].mean(), 1)) + "/" + str(
    #     round(run_data["Planned distance"].mean(), 1)) + ", Total deliveries = " + str(
    #     round(run_data["Total Deliveries"].mean(), 0)) + ", fatalities/1,000flighthours = " + str(
    #     run_data["fatalities_per_1000_flight_hours"].mean())
    #
    #       )

    t_per_shelter_category = run_data["Flighttime per shelter category"].mean()

    # for index,time in enumerate(t_per_shelter_category):
    #     cat_name = shelter_category_names[index]
    #     print("Time over {} = {} seconds".format(cat_name, time))

    #Average the risk map for all batch interations
    risk_map = np.zeros((run_data.Individual_risk_map[0][0].shape[0], run_data.Individual_risk_map[0][0].shape[1]))
    for i in range(n_iterations):
        risk_map += run_data.Individual_risk_map[i][0]
    risk_map = risk_map / n_iterations
    # Plot
    # risk_map = risk_map * (1 / risk_map.max())
    # plt.imshow(risk_map)
    # plt.colorbar()
    # plt.show()

    # Save data

    # run_data.to_csv("s3_5_P_off_t1.csv")
    # np.save('300x1000_with_CDRoff_pop_t1', risk_map)

    # These lines measure the unique headings present in the file.
    # headinglist = run_data["Heading list"][0]
    # headingarray = np.array(headinglist)
    # roundlist2 = headingarray.round(2)
    # unique_list = []
    #
    # for i in roundlist2:
    #     tpl = tuple(i)
    #     if not tpl in unique_list:
    #         unique_list.append(tpl)
    #
    # unique_array = np.array(unique_list)
    # counter = []
    # for i in unique_list:
    #     counter.append(np.count_nonzero(roundlist2 == np.array(i)))
    #
    # counterarray = np.array([counter])
    # counterarray = np.transpose(counterarray)
    # counter_and_uniques = np.append(counterarray, unique_array, axis=1)

    # print(run_data["Flown distance"] / run_data["Planned distance"])
    # print(run_data["Flown distance"].mean() / run_data["Planned distance"].mean())
    #
    # # Some checks
    agent_time = run_data["Time Climb and Descent"] + run_data["Cruise Time"] + run_data["Time Waiting"] + run_data[
        "Takeoff and landing time"]
    system_time = n_agents * max_steps

    if any(agent_time != system_time):
        raise Exception("Not all drone time is accounted for")
    #
    agent_flight_time = run_data["Time Climb and Descent"].mean() + run_data["Cruise Time"].mean() + run_data[
        "Takeoff and landing time"].mean()

    if round(sum(run_data["Flighttime per shelter category"].mean()) / agent_flight_time, 4) != 1:
        raise Exception("Not all flight time categorized in shelter categories")

    agent_air_time = agent_time - run_data["Time Waiting"] - run_data["Takeoff and landing time"]

    if any(agent_air_time != run_data["compute_flown_time_incl_unfinished"]):
        raise Exception("Error")
    #
    # #
    # plt.imshow(run_data["density_matrix_scaled_risk"][1][0].transpose())
    # plt.imshow(run_data["position_heatmap"][0][0].transpose(), alpha=0.7)
    # plt.title("No population density on water")
    # plt.show()
    # import copy
    # img = plt.imread("env/city_inputs/newyork/ny_corrected_map_scaled.png")
    # fig, ax = plt.subplots()
    # ax.imshow(img)
    # map = run_data["position_heatmap"][0][0].transpose()
    # map[map < 1] = np.nan
    # my_cmap = copy.copy(plt.cm.get_cmap('Reds'))
    # my_cmap.set_bad(alpha=0)
    # ax.imshow(map, cmap=my_cmap)
    # plt.title("ANN = {}".format(round(run_data["NNI_below_3"][0],2)))
    # save_loc = "env/city_inputs/newyork/ANN" + str(round(run_data["NNI_below_3"][0],2)) + ".png"
    # plt.savefig(save_loc)
    # plt.show()
    # filename = "data/v_impact.csv"
    # run_data.to_csv("data3/eval_fat_sens/shelter_9kg_v3.csv")
    # run_data.to_csv("data3/n  y/below_3_db.csv")
    run_data.to_csv("data/new_NNI_run_final_v1.csv")
    # plt.show()

