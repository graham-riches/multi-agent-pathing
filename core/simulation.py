"""
    @file simulation.py
    @brief quick test sim to try out the agent/arena and rendering
    @author Graham Riches
    @details
   
"""
import threading
from core.render_engine import Renderer
from core.command_line import CommandLine
from routing.a_star import AStar
from routing.managers.sequential import Sequential
from core.agent import *
from core.arena import Arena

BASE_TIME_STEP = 0.05
BASE_DPI = 40


# CLI thread
def cli_thread_func(arena: Arena, agents: list) -> None:
    cli = CommandLine(arena, agents)
    while True:
        cli.get_input()


# setup an arena
sim_arena = Arena(40, 40)

# create some agents and add them to the agents list
sim_agent = Agent(0, 0, BASE_TIME_STEP)
sim_agent_1 = Agent(5, 5, BASE_TIME_STEP)
sim_agent_1.set_kinematic_parameters(4, 4, 4)
sim_agent.set_kinematic_parameters(6, 6, 6)
sim_agents = list()
sim_agents.append(sim_agent)
sim_agents.append(sim_agent_1)

# setup the routing algorithm
algorithm = AStar(sim_arena, sim_agents)
algorithm.turn_factor = 2

# create the pathing algorithm
routing_manager = Sequential(sim_arena, sim_agents, algorithm)

# setup the renderer
renderer = Renderer(sim_arena, sim_agents, routing_manager, BASE_TIME_STEP, BASE_DPI)

# start the CLI thread
cli_thread = threading.Thread(target=cli_thread_func, args=(sim_arena, sim_agents))
cli_thread.start()


while True:
    renderer.render_arena()
    routing_manager.run_time_step()
    for agent_id, agent in enumerate(sim_agents):
        renderer.render_agent(agent_id)
    renderer.update()
