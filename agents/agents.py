# -*- coding: utf-8 -*-
"""
Specifies all agents except of the droneAgent
"""
import numpy as np
from mesa import Agent

# from droneAgentHelpers import DroneAgent




class ObstacleAgent(Agent):
    '''Plot the shelter factors at each location'''
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)




class PopulationDensity(Agent):
    '''
    Population density is modeled as agents to be able to visualize pop density. This can be considered a shortcoming of Mesa, because technically, Pop Density is not an agent at all.
    '''

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        if self.model.seed != None:
            np.random.seed(self.model.seed)
        # self.density = np.random.randint(10)
        self.density = 0
        self.changed = 0

    def colorchange(self):
        surroundings = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False, radius=2)

        for i in surroundings:  # trying to remove obstacle agents from possibilities, this doesnt work yet
            content = self.model.grid.get_cell_list_contents(i)
            for j in content:
                if type(j) is ObstacleAgent:
                    self.density = 7

        surroundings = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False, radius=1)

        for i in surroundings:  # trying to remove obstacle agents from possibilities, this doesnt work yet
            content = self.model.grid.get_cell_list_contents(i)
            for j in content:
                if type(j) is ObstacleAgent:
                    self.density = 9


'''Hub agent'''


class HubAgent(Agent):
    '''
    This is the class that specifies Hubs.
    '''
    def __init__(self, unique_id, model, unlimited_demand=False, demand_per_hour=None, demand_generation_interval="fixed"):
        """

        @param unique_id: all agents have an unique ID
        @param model: the model class is passed along here.
        @param unlimited_demand: whether or not demand is unlimited
        @param demand_per_hour: demand in tasks per hour
        @param demand_generation_interval: "fixed" or "poisson_process"
        """
        super().__init__(unique_id, model)


        # if n_orders == None:
        #     self.fixed_order_availability = False
        # else:
        #     self.fixed_order_availability = True
        self.fixed_order_availability = True
        self.orders = 0
        self.last_order = 0
        self.demand_generation_interval = demand_generation_interval

        assert demand_generation_interval == "fixed" or demand_generation_interval == "poisson" # must be either one of these two!

        if demand_generation_interval == "fixed":
            self.demand_timelist = np.arange(0, self.model.max_steps + 1, 3600 / demand_per_hour)
        elif demand_generation_interval == "poisson":
            self.demand_timelist = []
            demand_per_second = demand_per_hour / 3600
            self.waiting_times = np.random.poisson(demand_per_second, self.model.max_steps)
            for time, demand in enumerate(self.waiting_times):
                for i in range(0,demand):
                    # Poisson can also generate values > 1. In that case, the time is appended multiple times to the
                    # list In this way, we ensure that for large demand quantities, no demand is un-allocated if two
                    # units of demand are to be added at a time
                    self.demand_timelist.append(time)

        self.unlimited_demand = unlimited_demand
        if self.unlimited_demand == True and (self.fixed_order_availability == True):
            raise Exception("Unlimited demand and fixed amount of orders cant both be turned on")

        if self.model.seed is not None:
            np.random.seed(self.model.seed)

    def receiveorders(self):
        '''Method that adds tasks to the tasklist.'''
        current_time = self.model.schedule.steps

        while len(self.demand_timelist) > 0 and (current_time > self.demand_timelist[0]):
            # The while loop ensures that if a certain time is present multiple times in the demand-list, more units
            # of demand are added

            self.orders += 1
            self.model.total_demand += 1
            self.demand_timelist = np.delete(self.demand_timelist,0)


            # cellmates = self.model.grid.get_cell_list_contents([self.pos])
            # for cellmate in cellmates:
            #     if type(cellmate) is DroneAgent and cellmate.state == "waiting":
            #         print("Delivered package")
            #         cellmate.state = "to_customer"
            #         cellmate.target = random.choice(cellmate.deliver_pos)
            #         break

    def step(self):
        if self.unlimited_demand and self.orders < 1:
            raise Exception("Unlimited demand is on but a hub has no demand left.")
        elif (not self.unlimited_demand):
            self.receiveorders()


'''Delivery point agent'''


class DeliveryPointAgent(Agent):
    '''
    DeliveryPointAgent is the class for modeling DPs.
    '''


    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.deliveries = 0  # No. of packages delivered
