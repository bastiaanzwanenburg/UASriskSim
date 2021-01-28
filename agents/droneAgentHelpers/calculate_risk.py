'''
This method calculates the risk of the drone-agent at a location.
This is done by simulating a failure at this location and the impact that follows from it.
'''


import numpy as np
from agents.droneAgentHelpers.wind import wind
import math


def CalculateRisk(droneAgent, **kwargs):
    """
    This method calculates the risk of the drone-agent at a location.
    This is done by simulating a failure at this location and the impact that follows from it.

    @param droneAgent: class of drone agent
    @param kwargs: sx, sy, sz, vx, vy, vz can be passed along to this method.
    @return:
    """
    # Initialize initial state
    # If values are not set in calling function, then they are equal to the current agent's state

    # Because the drag is stochastic, the crash location is too. Therefore, we do a MC of max_risk_iterations. - BZ
    risk_map_this_step = np.zeros((droneAgent.model.width_risk, droneAgent.model.height_risk))
    i = 0

    CD = 0.7

    # Scale shelter map to the size of risk map

    planning = kwargs.get("planning",
                          True)  # if False, the computed risk is added to the total risk. if True the risk is only computed but not stored.

    sx = kwargs.get("sx", droneAgent.pos_m[0])
    sy = kwargs.get("sy", droneAgent.pos_m[1])
    sz = kwargs.get("sz", droneAgent.current_altitude)
    vx = kwargs.get("vx", droneAgent.v_horizontal * droneAgent.heading[0])
    vy = kwargs.get("vy", droneAgent.v_horizontal * droneAgent.heading[1])
    vz = kwargs.get("vz", droneAgent.v_vertical)

    # position = (sx, sy, sz)
    # speed = (vx, vy, vz)

    # MC run of risk calculation
    droneAgent.max_risk_iterations = 1000
    # wind_this_run = droneAgent.random.choices(wind, k=droneAgent.max_risk_iterations)

    # risk_from_db = droneAgent.model.riskDatabase.retrieve_risk(position, speed)
    risk_from_db = None
    if risk_from_db is None:
        droneAgent.model.risk_miss += 1
    else:
        droneAgent.model.risk_hit += 1
    # risk_from_db = None # TODO: fix this
    if risk_from_db is None:

        sx = kwargs.get("sx", droneAgent.pos_m[0])
        sy = kwargs.get("sy", droneAgent.pos_m[1])
        sz = kwargs.get("sz", droneAgent.current_altitude)
        vx = kwargs.get("vx", droneAgent.v_horizontal * droneAgent.heading[0])
        vy = kwargs.get("vy", droneAgent.v_horizontal * droneAgent.heading[1])
        vz = kwargs.get("vz", droneAgent.v_vertical)

        crash_locations, E_impact_database, v_impact_database = find_crash_locations(droneAgent, vx, vy, vz, sz)

        probability_checker = 0
        for position, probability_of_impact in crash_locations.items():
            probability_checker += probability_of_impact

            sx_impact = sx + position[0]
            sy_impact = sy + position[1]
            E_impact = E_impact_database[position]
            V_impact = v_impact_database[position]
            vx = V_impact[0]
            vy = V_impact[1]
            vz = V_impact[2]

            failure_loc = (sx, sy)
            impact_loc = (sx_impact, sy_impact)
            # print("Failure at {}, crash at {}".format(failure_loc, impact_loc))

            # Determine the impact-grid-points.
            impact_grid_x = int(round(
                sx_impact / (droneAgent.model.grid_size_m_x * droneAgent.model.width / droneAgent.model.width_risk),
                0))
            impact_grid_y = int(
                round(sy_impact / (
                            droneAgent.model.grid_size_m_y * droneAgent.model.height / droneAgent.model.height_risk),
                      0))

            if impact_grid_x >= droneAgent.model.width_risk or impact_grid_y >= droneAgent.model.height_risk or impact_grid_x < 0 or impact_grid_y < 0:
                # print("terminate at x,y = {}, {}".format(impact_grid_x, impact_grid_y))
                if planning:
                    # If CalculateRisk is being ran for path-finding, return only zeros NOTE: emergent behavior might
                    # be observed where paths around the edge of the map are "safer" as the risk=0 when it crashes
                    # outside of the map!!!
                    return 0
                else:
                    return

            # get sheltering factor
            ps = droneAgent.model.shelter_map_scaled_risk[impact_grid_x, impact_grid_y]


            if impact_grid_x == droneAgent.pos[0] and impact_grid_y == droneAgent.pos[1]:
                droneAgent.model.same_grid += 1
            else:
                droneAgent.model.other_grid += 1

            crash_dist = ((sy - sy_impact) ** 2 + (sx - sx_impact) ** 2) ** 0.5
            droneAgent.model.crash_dist.append(crash_dist)
            droneAgent.model.Eimp.append(E_impact)
            k = min(1, pow((droneAgent.model.beta / E_impact), 3 / ps))
            value = pow((droneAgent.model.alpha / droneAgent.model.beta), 0.5) * pow((droneAgent.model.beta / E_impact),
                                                                                     3 / (ps))
            pf = (1 - k) / (1 - 2 * k + value)

            risk_now = pf * droneAgent.model.density_matrix_scaled_risk[impact_grid_x, impact_grid_y]

            impact_pos = (impact_grid_x, impact_grid_y)
            droneAgent.model.crash_in_quantile[0] += probability_of_impact
            # if droneAgent.pos in droneAgent.model.below_1_positions:
            #     droneAgent.model.crash_in_quantile[1] += probability_of_impact
            #     # droneAgent.model.crash_in_quantile[2] += probability_of_impact
            #     # droneAgent.model.crash_in_quantile[3] += probability_of_impact
            #
            # if droneAgent.pos in droneAgent.model.below_2_positions:
            #     droneAgent.model.crash_in_quantile[2] += probability_of_impact
            #     # droneAgent.model.crash_in_quantile[3] += probability_of_impact
            #
            # if droneAgent.pos in droneAgent.model.below_3_positions:
            #     droneAgent.model.crash_in_quantile[3] += probability_of_impact


            # NOTE: risk of flying off the map is not included

            if 0 <= impact_grid_x < droneAgent.model.width_risk and 0 <= impact_grid_y < droneAgent.model.height_risk:
                # risk = pf * (droneAgent.model.impact_area / (droneAgentHelpers.model.grid_size_m_x * droneAgentHelpers.model.grid_size_m_y * droneAgentHelpers.model.width * droneAgentHelpers.model.height / droneAgentHelpers.model.width_risk / droneAgentHelpers.model.height_risk))

                # Here we calculate the possibility that a person inside this grid, dies. If A_d = 1m^2 and the
                # grid is 100m^2, there is a 1% probability that this person actually dies following the impact!!
                # Code to print impact angle
                v_horiz = (vx ** 2 + vy ** 2) ** 0.5
                if v_horiz == 0:
                    impact_angle = math.atan(abs(vz) / 0.00001)
                else:
                    impact_angle = math.atan(abs(vz) / v_horiz)

                radius_person = 0.2 # meters
                radius_uav = 0.2 # meters
                height_person = 1.7 # meters

                A_exp = np.pi * (radius_person + radius_uav) ** 2 * math.sin(impact_angle) + 2 * (radius_person + radius_uav) * (height_person + radius_uav) * math.cos(
                    impact_angle) # experiments show that A_exp is approximately 2x higher during cruise.

                risk_individual = probability_of_impact * pf * A_exp / droneAgent.model.area_risk_grid_m # TODO: check if the / area_risk_grid_m makes sense!!
                # droneAgent.model.impact_list.append(A_exp)
                # droneAgent.model.impact_angle_list.append(impact_angle)


                risk_map_this_step[impact_grid_x][impact_grid_y] += risk_individual

        if abs(probability_checker - 1) > 0.01:
            raise Exception("Woops, total probability = {}".format(probability_checker))

        # droneAgent.model.impact_aread the impact map from this step to the impact map of the drone
        avg_risk_map = risk_map_this_step
        # collective_risk = np.multiply(avg_risk_map, droneAgentHelpers.model.density_matrix_scaled_risk)
        if droneAgent.model.save_risk_maps:
            risk_to_use = avg_risk_map
        elif not droneAgent.model.save_risk_maps:
            collective_risk = np.sum(
                np.multiply(avg_risk_map, droneAgent.model.density_matrix_scaled_risk_area_grid_risk_m))
            risk_to_use = float(collective_risk)
        else:
            raise Exception("Unknown value")

        # droneAgent.model.riskDatabase.add_risk_to_database(position, speed, risk_to_use, store_to_file=False)

    else:
        risk_to_use = risk_from_db

    if not planning:
        if droneAgent.flight_phase == "climb" or droneAgent.flight_phase == "descend":
            risk_to_use = risk_to_use * droneAgent.model.failure_rate_climb_descend
        elif droneAgent.flight_phase == "cruise":
            risk_to_use = risk_to_use * droneAgent.model.failure_rate_cruise
        else:
            raise Exception("Unknown flight phase in risk computation!")

        if droneAgent.model.save_risk_maps:
            # For path-finding, we need to return the absolute risk-value, therefore, it is calculated here.
            # Only path-finding makes use of the "return" value of this formula, therefore, this is a safe option.
            droneAgent.risk_path_individual += risk_to_use
            return np.sum(np.multiply(risk_to_use, droneAgent.model.density_matrix_scaled_risk_area_grid_risk_m))
        else:
            # Risk is stored temporarily in the droneAgent object, and only added to the model totals after a drone
            # has completed a task

            droneAgent.collective_risk_temp += risk_to_use
            if droneAgent.flight_phase == "climb" or droneAgent.flight_phase == "descend":
                droneAgent.collective_risk_temp_climb_descent += risk_to_use
            elif droneAgent.flight_phase == "cruise":
                droneAgent.collective_risk_temp_cruise += risk_to_use
            else:
                raise Exception("Unknown flight phase in risk computation!")

            return risk_to_use

        # droneAgentHelpers.model.total_risk += avg_risk_map
    return risk_to_use


#
def find_crash_locations(droneAgent, vx_init, vy_init, vz_init, sz_init):
    '''
    This method finds the crash locations of a droneAgent, given an initial speed & altitude.
    It uses a database to do just-in-time computations of the required crash locations. It returns the crash locations relative to the (0,0) position.

    As the drone can have multiple possible crash locations given the stochasticity of wind & drag, a dictionary is returned with all crash locations, the impact energy, and the probability of crashing at that location.

    @param droneAgent: an object which is the drone agent in question. This is used to access the "model" class, as
    well as specific drone parameters (such as weight, frontal area, etc).
    @param vx_init: initial velocity in the x direction
    @param vy_init: intial velocity in the y direction
    @param vz_init: initial velocity in the z direction
    @param sz_init: initial position on the z-axis (= altitude)
    @return: two dictionaries (crash_locations_dict and E_impact_dict
    '''

    # Check if in database. If true: return. Else: add to database
    speed = (vx_init, vy_init, vz_init)

    # Try to find the crash-locations from the database.
    crash_locations_from_db, E_impact_locations_from_db, v_impact_from_db = droneAgent.model.crashLocationDatabase.retrieve_crash_locations(
        speed, sz_init)

    if type(crash_locations_from_db) == dict and type(E_impact_locations_from_db) == dict and type(v_impact_from_db) == dict:
        return crash_locations_from_db, E_impact_locations_from_db, v_impact_from_db
    # else is not necessary here as 'return' ends the function

    sx_start = 0
    sy_start = 0

    sz_start = sz_init
    vx_start = vx_init
    vy_start = vy_init
    vz_start = vz_init

    # Sensitivity study showed that reducing this to 0.2 or 0.1 has no influence.
    dt = 0.5

    n_iterations = droneAgent.model.max_risk_iterations

    # Using dictionaries for optimal performance
    crash_locations_dict = {}
    E_impact_dict = {}
    v_impact_dict = {}

    if droneAgent.model.include_wind:
        wind_this_simulation = droneAgent.random.choices(wind, k=n_iterations)
        droneAgent.model.max_risk_iterations = len(wind_this_simulation)
    else:
        wind_this_simulation = [[0, 0]]
        droneAgent.model.max_risk_iterations = 1
    k = 0
    while k < droneAgent.model.max_risk_iterations:
        # print("k = {}, max_risk_iter = {}".format(k, droneAgent.model.max_risk_iterations))
        if not droneAgent.model.include_wind:
            wind_speed = [0, 0]
        else:
            wind_speed = wind_this_simulation[k]
        k += 1
        if droneAgent.model.random_CD:
            CD = np.random.normal(loc=0.7, scale=0.1)
        else:
            CD = 0.7

        # The drag coefficient is bound between 0.4 and 1.1 to reduce outliers.
        CD_min = 0.4
        CD_max = 1.1
        CD = min(CD_max, CD)
        CD = max(CD_min, CD)

        CdRhoAMass = CD * droneAgent.frontal_area * droneAgent.model.air_density / (2 * droneAgent.mass)

        t = 0

        sx = sx_start
        sy = sy_start
        sz = sz_start
        vx = vx_start
        vy = vy_start
        vz = vz_start

        i = 0
        while sz > 0:

            t += dt
            i += 1
            if i > 1000:
                raise Exception("Seems like infinite loop...")

            # Acceleration at time t
            wind_speed_x = wind_speed[0] * (sz / 10) ** 0.3
            wind_speed_y = wind_speed[1] * (sz / 10) ** 0.3
            ax = CdRhoAMass * (vx - wind_speed_x) ** 2
            ay = CdRhoAMass * (vy - wind_speed_y) ** 2
            if vz > 0:
                az = - droneAgent.model.gravitational_constant - CdRhoAMass * vz ** 2
            if vz <= 0:
                az = - droneAgent.model.gravitational_constant + CdRhoAMass * vz ** 2

            # Position at time t + dt
            sx += vx * dt
            sy += vy * dt
            sz += vz * dt

            # Velocity at time t + dt
            if vx > wind_speed[0]:
                vx -= ax * dt
            else:
                vx += ax * dt

            if vy > wind_speed[1]:
                vy -= ay * dt
            else:
                vy += ay * dt

            vz += az * dt

        crash_location = (round(sx, 0), round(sy, 0))  # This rounding was verified to be valid
        # Three scenarios:
        # 1) v_impact is based on the model
        # 2) v_impact is based on v_terminal (found to be 71.75 m/s)
        # 3) v_impact is 40% higher than v_terminal
        droneAgent.model.vimp.append((vx ** 2 + vy ** 2 + vz ** 2)**0.5)

        if droneAgent.model.v_impact_scenario == 1:
            E_impact = 0.5 * droneAgent.mass * (vx ** 2 + vy ** 2 + vz ** 2)



        elif droneAgent.model.v_impact_scenario == 2:
            E_impact = 0.5 * droneAgent.mass * (71.75)**2
        elif droneAgent.model.v_impact_scenario == 3:
            E_impact = 0.5 * droneAgent.mass * (1.4*71.75)**2
        else:
            # Verification step for if an unknown scenario si given.
            raise Exception("Unknown scenario")

        if crash_location in E_impact_dict:
            # Often, there are subtle differences in the E_impact for the same crash location (Order of 1-3%). In
            # order to best account for this, the weighted average is taken of all E_impacts at this place.

            new_E_impact = (E_impact_dict[crash_location] * crash_locations_dict[crash_location] + E_impact) / (
                        crash_locations_dict[crash_location] + 1)
            if abs(E_impact_dict[crash_location] / new_E_impact - 1) > 1.1:
                # If the new E_impact is significantly different, something
                raise Exception("Same impact location but significantly different impact speeds")
            E_impact_dict[crash_location] = new_E_impact
        else:
            E_impact_dict[crash_location] = E_impact

        vimp = (vx, vy, vz)
        if crash_location not in v_impact_dict:
            v_impact_dict[crash_location] = vimp

        if crash_location in crash_locations_dict:
            crash_locations_dict[crash_location] += 1
        else:
            crash_locations_dict[crash_location] = 1

    # There are many crash_locations with a probability of < 5%. In the following, they are removed. This boosts
    # performance by 2-3x, while not significantly altering the results.

    factor = 1.0 / sum(crash_locations_dict.values())
    highest_value = max(crash_locations_dict.values())
    items_to_delete = []
    for p in crash_locations_dict:
        old_value = crash_locations_dict[p]
        crash_locations_dict[p] = crash_locations_dict[p] * factor
        if crash_locations_dict[p] < 0.05 and old_value < 0.05 * highest_value:
            items_to_delete.append(p)

    for item in items_to_delete:
        del crash_locations_dict[item]
        del E_impact_dict[item]

    factor = 1.0 / sum(crash_locations_dict.values())
    for p in crash_locations_dict:
        crash_locations_dict[p] = crash_locations_dict[p] * factor

    # A quick test
    for key in crash_locations_dict.keys():
        if key not in E_impact_dict:
            raise Exception("E_impact is missing!")

    droneAgent.model.crashLocationDatabase.add_crash_locations_to_database(speed, sz_init, crash_locations_dict,
                                                                           E_impact_dict, v_impact_dict)
    return crash_locations_dict, E_impact_dict, v_impact_dict

    # for item in items_to_delete:
    #     del crash_locations_dict[item]

    # input: drone dynamics, vx & vy pair

    # output: list with crash locations [ (relative position_tuple), probability]
    # drop those with probability < 0.01?
