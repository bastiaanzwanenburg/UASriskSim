# -*- coding: utf-8 -*-
"""
File which is accessed by run.py to set up the server which visualizes the model.
"""

from agents.agents import HubAgent, DeliveryPointAgent, ObstacleAgent, PopulationDensity
from agents.drone_agent import DroneAgent

'''Display setup'''
def agent_portrayal(agent):
    portrayal = {"Filled": "true"}

    if type(agent) is DroneAgent:
        portrayal["Shape"] = "circle"
        if agent.package == True:
            #If the agent has a package, make it larger
            portrayal["r"] = 1.5
        else:
            portrayal["r"] = 1

        if agent.hover == True:
            portrayal["Color"] = "black"
            portrayal["Layer"] = 4

        else:
            portrayal["Color"] = "grey"
            portrayal["Layer"] = 3

    elif type(agent) is HubAgent:
        portrayal["Shape"] = "circle"
        portrayal["Color"] = "green"
        portrayal["Layer"] = 2
        portrayal["r"] = agent.orders/10 + 1 #make the radius of the hub smaller for each package picked up.

    elif type(agent) is DeliveryPointAgent:
        portrayal["Shape"] = "circle"
        portrayal["Color"] = "blue"
        portrayal["Layer"] = 2
        portrayal["r"] = agent.deliveries/10 + 1 #make the radius of the delivery point larger for each package delivered.

    elif type(agent) is ObstacleAgent:
        portrayal["Color"] = "black"
        portrayal["Shape"] = "rect"
        portrayal["w"] = 1#np.random.randint(1,3)
        portrayal["h"] = 1#np.random.randint(1,3)
        portrayal["Layer"] = 1

    #"""below a piece of code which generates the population density, there are ten levels with each a different shade of red"""
    elif type(agent) is PopulationDensity:
        # The color is white (probability of fatality = 0) to red (pof = 1). This doesnt correct for pop density.
        portrayal["Shape"] = "rect"
        portrayal["Color"] = "green"
        portrayal["Layer"] = 0
        portrayal["w"] = 1
        portrayal["h"] = 1
        red_value = 255
        other_value = int(255 - (255 * agent.density))

        if 0 <= red_value <= 255 and 0 <= other_value <= 255:
            portrayal["Color"] = '#%02x%02x%02x' % (red_value, other_value, other_value)
        else:
            raise Exception("Not a valid color")


        # '''#ff1919, #ff3232, #ff4c4c, #ff6666, #ff7f7f, #ff9999, #ffb2b2, #ffcccc, #ffe5e5'''
        # if agent.density == 2.3:
        #     portrayal["Color"] = "#FF0000"
        # elif agent.density == 8:
        #     portrayal["Color"] = "#ff1919"
        # elif agent.density == 1.6:
        #     portrayal["Color"] = "#ff3232"
        # elif agent.density == 6:
        #     portrayal["Color"] = "#ff4c4c"
        # elif agent.density == 1.2:
        #     portrayal["Color"] = "#ff6666"
        # elif agent.density == 4:
        #     portrayal["Color"] = "#ff7f7f"
        # elif agent.density == 0.7:
        #     portrayal["Color"] = "#ff9999"
        # elif agent.density == 2:
        #     portrayal["Color"] = "#ffb2b2"
        # elif agent.density == 0.1:
        #     portrayal["Color"] = "#ffcccc"
        # elif agent.density == 0:
        #     portrayal["Color"] = "#ffe5e5"
        # elif agent.density == 10:
        #     portrayal["Color"] = "#696969	"
    
    return portrayal
