'''
The most important agent of the model is the DroneAgent. This file specifies it.
'''

import math
# import random
from math import *

import numpy as np
import portalocker
from mesa import Agent
from warnings import warn

import os
import pickle


from agents.agents import HubAgent, DeliveryPointAgent
from agents.droneAgentHelpers.calculate_risk import CalculateRisk
from agents.droneAgentHelpers.findAStarPath import find_a_star_path

class DroneAgent(Agent):

    # Initialize parameters for drone
    def __init__(self, unique_id, model, hub_pos, deliver_pos):

        '''
        Args:
            unique_id: Unique agent identifyer.
            pos: Position (in grid) of the drone
            pos_m: Position in meters of the drone
            altitude: Starting altitude
            state: to_customer or back_hub or waiting
            v_horizontal: Distance (meter) to move per step.
            hub_pos: position where the hub of the drone is
            deliver_pos: position where the drone to deliver to
            waypoints: the waypoints on the path to follow
            target: postion to move towards

            path: the remained waypoints on the path to follow (exclude waypoints flown)
            optimal_dis: the shortest dis to target
            planned_dis: the distance flown and to fly in the flight plan
            flown_dis: the distance flown by the drone
        '''
        super().__init__(unique_id, model)

        self.pos = hub_pos  # Position (in grid) of the drone
        self.pos_m = (
            hub_pos[0] * self.model.grid_size_m_x,  # Position (in meter) of the drone
            hub_pos[1] * self.model.grid_size_m_y)

        # Vehicle parameters
        self.cruise_altitude = self.model.cruise_altitude  # Starting altitude
        self.vertical_speed = 5  # m/s
        self.current_altitude = 0
        self.cruise_speed = 10  # Drone v_horizontal = 20m/s
        self.mass = 9 * self.model.mass_correction_factor
        self.frontal_area = 0.04 # A_d of the drone [m^2]
        self.collective_risk_temp = 0
        self.collective_risk_temp_climb_descent = 0
        self.collective_risk_temp_cruise = 0

        self.v_horizontal = 0
        self.v_vertical = 0

        self.m_route = []
        self.m_target_route = []

        self.heading = (0, 0)  # Heading of the drone
        self.hub_pos = hub_pos  # position where the hub of the drone is
        self.deliver_pos = deliver_pos  # position where the drone to deliver to
        self.target = None  # postion to move towards
        self.state = 'waiting'
        self.hover = False
        # self.state = 'to_customer'
        self.package = True
        self.flight_plan = []
        self.path = []

        self.planned_path = []
        self.flown_path = []

        if self.cruise_speed > self.model.grid_size_m_x or self.cruise_speed > self.model.grid_size_m_y:
            print("speed: {}, grid_x = {}, grid_y = {}".format(self.cruise_speed, self.model.grid_size_m_x,
                                                               self.model.grid_size_m_y))
            raise Exception("Speed larger than grid-size, this leads to issues!")

        # Risk initialization
        self.max_risk_iterations = self.model.max_risk_iterations
        self.risk_path_individual = np.zeros((self.model.width_risk, self.model.height_risk))
        self.risk_path_collective = np.zeros((self.model.width_risk, self.model.height_risk))  # not uesed yet
        if self.model.seed != None:
            np.random.seed(self.model.seed)

        # Metrics
        self.total_flown_time_test = 0
        self.total_flown_time = 0

        self.optimal_dis_total_completed = 0  # Optimal distance of all completed flights
        self.planned_dis_total_completed = 0  # Planned distance of all completed flights
        self.flown_dis_total_completed = 0  # Flown distance of all completed flights

        self.optimal_dis_current = 0
        self.planned_dis_current = 0
        self.flown_dis_current = 0

        self.optimal_dis_total = 0  # Optimal distance of all flights (including unfinished flights)
        self.planned_dis_total = 0  # Planned distance of all flights (including unfinished flights)
        self.flown_dis_total = 0  # Flown distance of all flights (including unfinished flights)

        self.flown_time_current = 0
        self.total_flown_time_incl_unfinished = 0
        self.hover_time = 0
        self.impact_position = (0, 0)

    '''Behaviour of drone for each steps: the drone will wait in the hub until receiving order, and start from hub to deliver point and then come back to (the same) hub.'''

    def step(self):
        # The following lines are purely for logging what is going on
        if self.state == 'waiting':
            self.model.time_waiting += 1
        elif self.state == 'to_customer' or self.state == 'back_hub':
            # Log where the drone is flying
            current_category = self.model.shelter_category_scaled[self.pos]
            self.model.flight_time_shelter_category[current_category] += 1

            # Log time spent in cruise / takeoff / climb/descend
            if self.current_altitude == self.cruise_altitude:
                self.model.time_cruise += 1
            elif self.current_altitude == 0 and self.pos == self.target:
                self.model.time_takeoff_landing += 1
            else:
                self.model.time_climb_descent += 1
        else:
            raise Exception("Drone is in unknown state")

        # Calculate the risk of this step
        if self.state == 'to_customer' or self.state == 'back_hub':
            if self.current_altitude > 0:
                # When the altitude = 0, there is no risk, hence the > 0
                CalculateRisk(self, planning=False)

            # The drone can not move horizontally and vertically at the same time, verify what is going wrong

            if self.v_horizontal == self.cruise_speed is abs(self.v_vertical) == self.vertical_speed:
                raise Exception("We have both horizontal and vertical v_horizontal. That's not possible!")

        elif self.state == 'waiting':
            if self.v_horizontal != 0 or self.v_vertical != 0:
                # Speed must be 0 while waiting, verify this
                raise Exception("Non-zero v_horizontal while waiting...")

        # If going to a customer, go through all steps required to ultimately deliver a package.
        if self.state == 'to_customer':
            # If not at the target & we don't have a flight-plan yet, generate one.
            if (self.pos != self.target) and len(self.flight_plan) == 0:
                self.start_new_mission()

            # Check if the drone is at the altitude it should be. If that's the case, the drone can move, otherwise,
            # the climb_and_descend-method makes sure it travels in the direction of the desired altitude
            is_at_target_altitude = self.climb_and_descent()

            if is_at_target_altitude:
                self.v_horizontal = self.cruise_speed

                # If the drone is at the target of the mission, deliver the package. Otherwise, move.
                if self.pos == self.target:
                    self.arrival_sequence()
                    self.DeliverPackage()
                    self.state = 'back_hub'
                    self.target = self.hub_pos
                else:
                    self.move()
            else:
                self.v_horizontal = 0
                # self.model.time_climb_descent += 1

        # If the goal is to go back to the hub, go through all steps required to do so.
        elif self.state == 'back_hub':
            self.target = self.hub_pos
            # If not at the target & we don't have a flight-plan yet, generate one.
            if (self.pos != self.target) and len(self.flight_plan) == 0:
                self.start_new_mission()

            is_at_target_altitude = self.climb_and_descent()

            if is_at_target_altitude:
                self.v_horizontal = self.cruise_speed
                if self.pos == self.target:
                    if self.flight_phase != "arrived":
                        raise Exception("Flight phases are not correct")
                    self.arrival_sequence()
                    self.state = 'waiting'
                else:
                    self.move()
            else:
                self.v_horizontal = 0

        elif self.state == "waiting":
            # The drone can only be waiting at the hub. Check if this is the case.
            cellmates = self.model.grid.get_cell_list_contents([self.pos])
            if not cellmates or not any(isinstance(x, HubAgent) for x in cellmates):
                raise Exception("ERROR, DRONE IS WAITING SOMEWHERE WITHOUT CELLMATES / NO HUB AS CELLMATE")

            # The following code makes sure the drone is assigned an package.
            for cellmate in cellmates:
                if type(cellmate) is HubAgent:
                    # Check if the hub has outstanding orders. If so, pick up the task.
                    if cellmate.orders > 0:
                        if not cellmate.unlimited_demand:
                            cellmate.orders -= 1
                        self.state = "to_customer"
                        self.model.total_pickups += 1
                        self.target = self.random.choice(self.deliver_pos)
                        # Break the execution of the for-loop. This isn't strictly necessary, as there can't be more 
                        # than 1 hub at each grid-location. However, it prevents the need of looping through all 
                        # other cellmates, which speeds things up a bit. 
                        break

    def arrival_sequence(self):
        '''Method for the arrival sequence. Primary objective is to store statistics of the past flight.'''
        self.flight_plan.clear()

        self.model.risk_map_individual += self.risk_path_individual
        self.model.collective_risk += self.collective_risk_temp
        self.model.collective_risk_cruise += self.collective_risk_temp_cruise
        self.model.collective_risk_climb_descend += self.collective_risk_temp_climb_descent

        self.collective_risk_temp = 0
        self.collective_risk_temp_cruise = 0
        self.collective_risk_temp_climb_descent = 0

        self.risk_path_individual = np.zeros((self.model.width_risk, self.model.height_risk))

        self.v_horizontal = 0
        self.v_vertical = 0

        self.total_flown_time = self.total_flown_time_incl_unfinished
        self.total_flown_time_test += self.flown_time_current

        self.flown_dis_total_completed += self.flown_dis_current
        self.optimal_dis_total_completed += self.optimal_dis_current
        self.planned_dis_total_completed += self.planned_dis_current
        #
        self.flown_dis_current = 0
        self.optimal_dis_current = 0
        self.planned_dis_current = 0
        self.flown_time_current = 0

    def start_new_mission(self):
        '''Whenever a drone gets a new task, this method is triggered. Amongst others, it finds a flight-plan and calculates some statistics.'''
        self.flight_plan = find_a_star_path(self, self.pos, self.target, self.model.risk_optimal)
        # print("Flight to {} via {}".format(self.target, self.flight_plan))

        # Calculate flying distance in flight plan
        planned_dis_this_flight = 0
        for i in range(0, len(self.flight_plan) - 1):
            planned_dis_this_flight += (

                pow(pow(self.model.grid_size_m_x * (self.flight_plan[i][0] - self.flight_plan[i + 1][0]), 2) +
                    pow(self.model.grid_size_m_y * (self.flight_plan[i][1] - self.flight_plan[i + 1][1]), 2),
                    0.5))

        self.flight_plan.pop(0)  # remove the first one, which is the current pos

        self.planned_dis_total += planned_dis_this_flight
        self.planned_dis_current += planned_dis_this_flight
        # self.planned_dis += planned_dis Do this after a flight!
        optimal_dis_this_flight = (
            pow(pow(self.model.grid_size_m_x * (self.flight_plan[-1][0] - self.flight_plan[0][0]), 2) +
                pow(self.model.grid_size_m_y * (self.flight_plan[-1][1] - self.flight_plan[0][1]), 2),
                0.5))
        self.optimal_dis_total += optimal_dis_this_flight
        self.optimal_dis_current += optimal_dis_this_flight

        if optimal_dis_this_flight > 1.001 * planned_dis_this_flight:
            # added 1.001 for rounding errors in straight paths.
            raise Exception("Optimal distance longer than planned distance")

    def detect_collissions(self):
        """
        A simple implementation of CD&R was implemented. However, it was decided to not use this the thesis.
        It returns True if the drone is in conflict, otherwise, it returns False
        @rtype: Boolean
        """
        #

        if self.model.conflictdetection == False:
            return False

        surroundings = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False, radius=1)
        collission_detected = False

        for i in surroundings:  # trying to remove obstacle agents from possibilities, this doesnt work yet
            content = self.model.grid.get_cell_list_contents(i)
            for j in content:
                # Check if there is another agent that is a drone
                if type(j) is DroneAgent:
                    if j.state == 'back_hub' or j.state == 'to_customer':
                        if not j.hover:
                            # print('collission detected')
                            return True
        return collission_detected

    def climb_and_descent(self):
        """
        This function handles the climbing & descend of drones. It checks whether the drone is at the altitude it should be. If yes, it returns True (signalling to the code that it can go on with the next step). Otherwise, it signals False
        @rtype: Boolean
        """

        if self.pos != self.target:
            # If not at the target, we should climb, set climbing v_horizontal
            if self.current_altitude >= self.cruise_altitude:
                if self.current_altitude > self.cruise_altitude:
                    print("Warning: flying {} meters above cruise altitude".format(
                        self.current_altitude - self.cruise_altitude))
                self.v_vertical = 0
                self.flight_phase = "cruise"
                return True

            elif (self.current_altitude + self.vertical_speed) >= self.cruise_altitude:
                self.current_altitude = self.cruise_altitude
                self.v_vertical = self.vertical_speed
                self.flown_time_current += 1
                self.total_flown_time_incl_unfinished += 1
                self.flight_phase = "climb"
                return False

            else:
                self.current_altitude += self.vertical_speed
                self.v_vertical = self.vertical_speed
                self.flown_time_current += 1
                self.total_flown_time_incl_unfinished += 1
                self.flight_phase = "climb"
                return False

        elif self.pos == self.target:
            # If the drone can reach an altitude of 0 in this step, just fix it to 0
            if self.current_altitude <= 0:
                if self.current_altitude < 0:
                    print("Warning: flying {} meters below destination altitude".format(
                        self.current_altitude))
                self.v_vertical = 0
                self.flight_phase = "arrived"
                return True

            elif self.current_altitude <= 0 + self.vertical_speed:
                self.current_altitude = 0
                self.v_vertical = -self.vertical_speed
                self.flown_time_current += 1
                self.total_flown_time_incl_unfinished += 1
                self.flight_phase = "descend"
                return False

            else:
                self.current_altitude -= self.vertical_speed
                self.v_vertical = -self.vertical_speed
                self.flown_time_current += 1
                self.total_flown_time_incl_unfinished += 1
                self.flight_phase = "descend"
                return False


    # Move forward, update dynamics
    def move(self):
        '''
        Moves the drone both virtually (self.pos_m) and physically (self.pos) if needed.
        @return: True if destination is reached
        '''

        # flown time count +1 and calculate risk

        self.flown_time_current += 1
        self.total_flown_time_incl_unfinished += 1

        # Set path (list of waypoints) of the drone to fly
        if len(self.path) == 0:
            self.path = self.flight_plan[:]
            self.planned_path = self.flight_plan[:]

        self.findNextVirtualStep()

        # check whether reached the next path waypoints (reached: distance within one step move (in meters))
        if self.dis_to_target > self.v_horizontal / 2:
            self.moveVirtualStep()

        else:
            # Move drone to the next waypoint in grid. Check if there are any collissions.
            if not self.detect_collissions():
                self.hover = False # This code is a reminder of the CD&R implementation, and not important for functioning of the current code.

                # Move to next point on grid AND next point on virtual (m) grid
                # NOTE: Because moves can also be diagonal, it is impossible to design the model in such a way that it always "perfectly" reaches the next point.

                next_point = (int(round((self.pos_m[0] / self.model.grid_size_m_x), 0)),
                              int(round((self.pos_m[1] / self.model.grid_size_m_y), 0)))

                # Two verification steps to ensure the integrity of the move function.
                if next_point != self.path[0]:
                    raise Exception("Moving, but not to the next point in the path")

                if self.pos == next_point:
                    raise Exception("Trying to move a drone to the same point as it already is!")

                # Move one step on the real grid
                self.model.grid.move_agent(self, next_point)
                self.model.position_heatmap[next_point] += 1
                self.path.pop(0)

                # Also move one step on the virtual grid.
                if self.pos != self.target:
                    self.findNextVirtualStep()
                    self.moveVirtualStep()

                # check if reached target
                if len(self.path) == 0:
                    return True
            else:
                self.hover = True
                self.hover_time += 1

    def findNextVirtualStep(self):
        """
        Based on the goal of the next step, the drone moves one step on the virtual grid. This is done by storing some variables, which are then accessed in subsequent methods.
        @rtype: None

        """
        self.next_target = self.path[0]
        self.next_target_m = (self.next_target[0] * self.model.grid_size_m_x,
                              self.next_target[1] * self.model.grid_size_m_y)

        # Calculate next position of drone after one step move (in meters)
        self.diff_to_next_m_x = self.next_target_m[0] - self.pos_m[0]
        self.diff_to_next_m_y = self.next_target_m[1] - self.pos_m[1]
        self.dis_to_target = pow(pow(self.diff_to_next_m_x, 2) + pow(self.diff_to_next_m_y, 2), 0.5)

    def moveVirtualStep(self):
        '''
        Given that self.dis_to_target and self.diff_to_next_m_x &..y are set (that is done by findNextVirtualStep), it moves to the next virtual step
        @return: None, but changes self.pos_m to the next virtual position
        '''

        # Calculate heading of drone
        self.heading = (self.diff_to_next_m_x / self.dis_to_target,
                        self.diff_to_next_m_y / self.dis_to_target)

        # self.model.heading_list.append(self.heading)
        # Calculate next position (in meters)

        self.next_pos_m = (self.pos_m[0] + self.v_horizontal * self.heading[0],
                           self.pos_m[1] + self.v_horizontal * self.heading[1])

        self.m_route.append(self.next_pos_m)
        self.m_target_route.append(self.next_target_m)

        delta_distance = ((self.next_pos_m[0] - self.pos_m[0]) ** 2 + (
                self.next_pos_m[1] - self.pos_m[1]) ** 2) ** 0.5

        self.flown_dis_current += delta_distance

        self.flown_dis_total += delta_distance

        if abs(pow(pow(self.pos_m[0] - self.next_pos_m[0], 2) + pow(self.pos_m[1] - self.next_pos_m[1], 2),
                   0.5) - self.v_horizontal) < 0.01 * self.v_horizontal:
            self.pos_m = self.next_pos_m
            # self.
        else:
            raise Exception("Next step is not equal to distance")

        # Calculate flown distance

    # Deliver package at delivery point
    def DeliverPackage(self):
        '''
        Triggered whenever a drone arrives at the end of the flight-plan, and delivers  pacakge.
        @rtype: None
        '''

        # Tries to deliver a package
        # If there's a delivery point in the same cell, deliver the package there.
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            for cellmate in cellmates:
                if type(cellmate) is DeliveryPointAgent:
                    # print("Delivered package")
                    cellmate.deliveries += 1
                    self.model.total_deliveries += 1
                    self.package = False
                    self.state = 'back_hub'
                    self.target = self.hub_pos
        if self.package:
            raise Exception("Delivering package failed")


