# UAS Risk Sim

UASriskSim is an agent-based model that simulates the Third Party Risk (TPR) following from Urban UAS operations for the cities of Delft, New York and Paris. It makes use of state-of-the-art methods of computing the TPR. The model was developed by Bastiaan Zwanenburg (b.j.v.zwanenburg@student.tudelft.nl) as part of a Master Thesis at TU Delft. The model is based on Mesa, an open-source ABM framework (https://mesa.readthedocs.io/en/stable/). 

## How to use the model

Here, a quick instruction is presented of how the model should be used. For the full documentation, please see: https://htmlpreview.github.io/?https://github.com/bastiaanzwanenburg/UASriskSim/blob/main/html/github_version/index.html

The model can be run by running one of two files: model.py (no visualization, faster) or run.py (includes visualization, slower). The model consists of three components: the agents and their interactions, the risk computation, and the environment.

### Agents & their interactions

The model features three types of agents: UAS, Delivery Points, and Hubs. 

### Environment

The model generates an environment based on pictures of the city topology and population density. The topology is used to generate sheltering factors. The population density is used to do population density.

Example of risk in New York:
![Image of risk in New York](https://github.com/bastiaanzwanenburg/UASriskSim/blob/main/html/riskNY.png)
  
