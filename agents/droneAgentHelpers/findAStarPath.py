'''
Find the (Risk)A* path of a route.
'''


import heapq
import numpy as np
from agents.droneAgentHelpers.calculate_risk import CalculateRisk


class Node:
    """
    A node class for A* Pathfinding
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0  # g = actual cost of reaching this node
        self.h = 0  # h = heuristic, used for determining which node to visit next
        self.f = 0  # f = g + h

    def __eq__(self, other):
        return self.position == other.position

    def __repr__(self):
        return f"{self.position} - g: {self.g} h: {self.h} f: {self.f}"

    # def __hash__(self):
    #     return int(self.position[0] + self.position[1])

    # defining less than for purposes of heap queue
    def __lt__(self, other):
        return self.f < other.f

    # defining greater than for purposes of heap queue
    def __gt__(self, other):
        return self.f > other.f

def find_a_star_path(droneAgent, current_pos, target_pos, risk_optimal, verbose=False):
    '''
    This method finds the shortest path between two locations. The data-structures have been optimized, which led to a 200x performance reduction in our case.
    Furthermore, it has the possibility to store the path in a database.
    @param droneAgent:
    @param current_pos: starting position of flight-plan
    @param target_pos: target position of flight-plan
    @param risk_optimal: "risk_optimal" calculates risk at each position. "shelter_and_pop_density" bases risk on grid directly below the drone. False leads to shortest-path.
    @param verbose:
    @return:
    '''

    from datetime import datetime
    startTime = datetime.now()

    path_from_db = droneAgent.model.routeDatabase.retrieve_path(current_pos, target_pos)
    # path_from_db = None # TODO: fix this
    if path_from_db != None:
        if verbose:
            print("Retrieved path from DB")
        return path_from_db[:]
    else:
        if verbose:
            print("Path from {} to {} not yet in DB".format(current_pos, target_pos))

    if (current_pos in droneAgent.model.obstacle_list) or (target_pos in droneAgent.model.obstacle_list):
        raise Exception("ERROR: CURRENT OR TARGET IN OBSTACLE LIST!")

    # Initialize start- and end-nodes with zero cost
    start_node = Node(None, current_pos)
    start_node.g = start_node.h = start_node.f = 0.0

    end_node = Node(None, target_pos)
    end_node.g = end_node.h = end_node.f = 0.0

    # Initialize open- and closed list
    # open_list = []
    # closed_list = []

    # Heapify the open_list and Add the start nodex
    # heapq.heapify(open_list)
    # heapq.heappush(open_list, start_node)

    open_list = [start_node]
    closed_set = set()
    open_dict = {start_node.position: start_node.g}

    # MAX ITERATIONS
    outer_iterations = 0
    max_iterations = (droneAgent.model.width * droneAgent.model.height) ** 2

    # As long as there are "open" nodes, we continue A*.
    while len(open_list) > 0:
        outer_iterations += 1
        if verbose and (outer_iterations % 2000 == 0):
            print("Iteration {}, closed node list {}, open node list {}, time={}".format(outer_iterations, len(closed_set), len(open_list), datetime.now() - startTime))
        if outer_iterations > max_iterations:
            # if we hit this point return the path such as it is
            # it will not contain the destination
            closed_map = np.zeros((droneAgent.model.width, droneAgent.model.height))
            for i in closed_set:
                closed_map[i] = 1
            import matplotlib.pyplot as plt
            plt.imshow(closed_map)
            # plt.title("len closed list = {}".format(len(closed_list)))
            plt.show()
            raise Exception("giving up on pathfinding too many iterations")

        # Find node with the lowest cost F
        current_node = heapq.heappop(open_list)

        if current_node.position in closed_set:
            continue
        closed_set.add(current_node.position)

        # IF current node is THE GOAL, go through all parent nodes of the current node.
        if current_node == end_node:
            path = []
            cost_g = 0
            cost_f = 0
            cost_h = 0
            current = current_node
            while current is not None:
                path.append(current.position)
                cost_g += current.g
                cost_f += current.f
                cost_h += current.h
                current = current.parent
            if verbose:
                print("Path found, f = {}, g = {}, h = {}".format(current_node.f, current_node.g, current_node.h))
                print("Found path in {} sec, closed list size = {}".format(datetime.now() - startTime,
                                                                           len(closed_set)))
            # print("Path: {}".format(path[::-1]))
            path = path[::-1]  # Reverse order and return path.
            droneAgent.model.routeDatabase.add_to_database(origin=current_pos, target=target_pos, path=path,
                                                     store_to_file=False)

            #
            return path

        # Generate children
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1),
                             (1, 1)]:  # Adjacent squares
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # If the node goes outside the range, skip it.
            if node_position[0] > (droneAgent.model.width - 1) or node_position[0] < 0 or node_position[1] > (
                    droneAgent.model.height - 1) or node_position[1] < 0:
                continue

            length_step = sum(map(abs, new_position)) ** 0.5
            # this returns 2 if a step is diagonal, and 1 if it is to an adjacent square

            # Make sure that node is walkable
            if node_position in droneAgent.model.obstacle_list:
                continue

            child = Node(current_node, node_position)

            # # Loop through children
            # for child in children:
            add_to_open = True

            # if child in closed_list:
            #     # If child is already closed, skip it.
            #     add_to_open = False
            #     continue

            # child.g = current_node.g + 1 # This is normal A* planning
            if risk_optimal == "pop_density":
                cost_step = droneAgent.model.density_matrix_scaled[child.position[0], child.position[
                    1]]  # Cost of moving to a node = the risk of that node * a multiplication factor if it is diagonal.

            elif risk_optimal == "calculate_risk":
                parent_node = child.parent
                parent_position = parent_node.position
                new_position = child.position

                dif_x = new_position[0] - parent_position[0]
                dif_y = new_position[1] - parent_position[1]
                dis_to_new_position = (dif_x ** 2 + dif_y ** 2) ** 0.5

                heading_new_position = (dif_x / dis_to_new_position, dif_y / dis_to_new_position)
                middle_point_x = (new_position[0] + parent_position[0]) / 2
                middle_point_y = (new_position[1] + parent_position[1]) / 2
                one_third_point_x_m = parent_position[0] + 1 / 3 * dif_x * droneAgent.model.grid_size_m_x
                one_third_point_y_m = parent_position[1] + 1 / 3 * dif_y * droneAgent.model.grid_size_m_y
                two_third_point_x_m = parent_position[0] + 2 / 3 * dif_x * droneAgent.model.grid_size_m_x
                two_third_point_y_m = parent_position[1] + 2 / 3 * dif_y * droneAgent.model.grid_size_m_y

                next_point_x_m = new_position[0] * droneAgent.model.grid_size_m_x
                next_point_y_m = new_position[1] * droneAgent.model.grid_size_m_y

                middle_point_x_m = middle_point_x * droneAgent.model.grid_size_m_x
                middle_point_y_m = middle_point_y * droneAgent.model.grid_size_m_y

                vx_new = droneAgent.cruise_speed * heading_new_position[0]
                vy_new = droneAgent.cruise_speed * heading_new_position[1]

                if droneAgent.model.risk_path_planning_intervals == 1:
                    risk_map_individual = CalculateRisk(droneAgent, sx=next_point_x_m, sy=next_point_y_m,
                                                        sz=droneAgent.cruise_altitude, vx=vx_new, vy=vy_new,
                                                        planning=True)
                elif droneAgent.model.risk_path_planning_intervals == 2:
                    risk_map_individual_1 = CalculateRisk(droneAgent, sx=next_point_x_m, sy=next_point_y_m,
                                                          sz=droneAgent.cruise_altitude, vx=vx_new, vy=vy_new,
                                                          planning=True)
                    risk_map_individual_2 = CalculateRisk(droneAgent, sx=middle_point_x_m, sy=middle_point_y_m,
                                                          sz=droneAgent.cruise_altitude, vx=vx_new, vy=vy_new,
                                                          planning=True)
                    risk_map_individual = (risk_map_individual_1 + risk_map_individual_2) / 2.0

                elif droneAgent.model.risk_path_planning_intervals == 3:
                    risk_map_individual_1 = CalculateRisk(droneAgent, sx=one_third_point_x_m, sy=one_third_point_y_m,
                                                          sz=droneAgent.cruise_altitude, vx=vx_new, vy=vy_new,
                                                          planning=True)
                    risk_map_individual_2 = CalculateRisk(droneAgent, sx=two_third_point_x_m, sy=two_third_point_y_m,
                                                          sz=droneAgent.cruise_altitude, vx=vx_new, vy=vy_new,
                                                          planning=True)
                    risk_map_individual_3 = CalculateRisk(droneAgent, sx=next_point_x_m, sy=next_point_y_m,
                                                          sz=droneAgent.cruise_altitude, vx=vx_new, vy=vy_new,
                                                          planning=True)

                    risk_map_individual = (risk_map_individual_1 + risk_map_individual_2 + risk_map_individual_3) / 3.0
                else:
                    raise Exception("Not supported")



                cost_step = risk_map_individual * 1000  # TODO: check this formula

                # if child.g < droneAgent.model.min_risk:
                # print("Cost is {}, but min_risk is {}".format(child.g, droneAgent.model.min_risk))


            elif risk_optimal == "shelter_and_pop_density":
                ps = droneAgent.model.shelter_map_scaled[child.position[0], child.position[1]]
                # Impact kinetic energy
                v_terminal = 50
                Eimp = 0.5 * 3 * v_terminal ** 2
                alpha = 32000  # impact energy required for a fatality probability of 50% when ps = 6
                beta = 34  # impact energy needed to cause a fatality when ps approaches zero

                # Falality probability

                k = min(1, (beta / Eimp) ** (3 / ps))
                # k = min(1, pow((beta / Eimp), 3 / ps))
                value = ((alpha / beta) ** 0.5) * ((beta / Eimp) ** (3 / ps))
                # value = pow((alpha / beta), 0.5) * pow((beta / Eimp), 3 / (ps))
                pf = (1 - k) / (1 - 2 * k + value)
                value = pf * droneAgent.model.density_matrix_scaled[child.position[0], child.position[1]]

                # droneAgent.model.a_star_value.append(value)
                # value = 1 + 10*value
                cost_step = value  # add_value is already incorporated in heuristic.
            else:
                cost_step = 1

            distance_to_goal = (((child.position[0] - end_node.position[0]) ** 2) + (
                        (child.position[1] - end_node.position[1]) ** 2)) ** 0.5
            # distance_to_goal = pow((pow((child.position[0] - end_node.position[0]), 2) + pow(
            #     (child.position[1] - end_node.position[1]), 2)), 0.5)

            child.g = length_step * cost_step + current_node.g
            child.h = droneAgent.model.min_risk * distance_to_goal
            child.f = child.g + child.h

            pos = child.position
            add_to_open = pos not in closed_set and (pos not in open_dict or open_dict[pos] > current_node.g)
            if add_to_open:
                heapq.heappush(open_list, child)
                open_dict[current_node.position] = current_node.g

            # TODO THiS CHECK SHOULD BE ON SOMETIMES
            # if droneAgent.model.min_risk > length_step * cost_step:
            # raise Exception(
            #     "Heuristic not admissisble, min_risk={}, child.g={}".format(droneAgent.model.min_risk, child.g))

            # print("Child.f: {}, type: {}".format(child.f, type(child.f)))
            # next((x for open_node in open_list if x.position == child.position), None)

            # add_to_open = True
            # for open_node in open_list:
            #     if child.position == open_node.position:
            #         if child.f >= open_node.f:
            #             add_to_open = False
            #             break
            # if add_to_open == True:
            #     heapq.heappush(open_list, child)

            # if len([open_node for open_node in open_list if child == open_node and child.f >= open_node.f]) > 0:
            #     add_to_open = False
            #     continue

            # filtered_open_nodes = (open_node for open_node in open_list if child == open_node)
            # open_node = next(filtered_open_nodes, None)
            #
            # while open_node:
            #     # this unit-test can be turned on sometimes
            #     # if child.h != open_node.h:
            #     #     raise Exception("Same position but different heuristic, wtf")
            #     if child.g >= open_node.g: #TODO check if this strict equality is justified.
            #         add_to_open = False
            #         break
            #     else:
            #         open_list.remove(open_node)
            #         open_node = next(filtered_open_nodes, None)
            #
            # # for open_node in open_list:
            # #     if child == open_node:
            # #         if child.f >= open_node.f:
            # #             #Skipping nodes with equal cost G dramatically increases v_horizontal! (Due to the rectangular map structure, many nodes have equal cost G).
            # #             add_to_open = False
            # #             break
            # #         elif child.f < open_node.f:
            # #             open_list.remove(open_node)
            # if add_to_open == True:
            #     heapq.heappush(open_list, child)

            # open_node = next((x for x in open_list if x.position == child.position), False)
            # while open_node != False:
            #     if child.f > open_node.f:
            #         add_to_open = False
            #         open_node = False
            #     elif child.f < open_node.f:
            #         open_list.remove(open_node)
            #         open_node = next((x for x in open_list if x.position == child.position), None)
            #         if open_node == False:
            #             break
            #     else:
            #         open_node = None
            # if add_to_open:
            #     open_list.append(child)