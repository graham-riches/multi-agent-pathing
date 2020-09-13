"""
    @file simulation_test.py
    @brief quick test sim to try out the agent/arena and rendering
    @author Graham Riches
    @details
   
"""
import time
import threading
from render_engine import Renderer
from command_line import CommandLine
from agent import Agent, AgentCoordinates, AgentState
from arena import Arena

BASE_TIME_STEP = 0.05
BASE_DPI = 20


# CLI thread
def cli_thread_func(arena: Arena, agents: list) -> None:
    cli = CommandLine(arena, agents)
    while True:
        cli.get_input()


# setup an arena
sim_arena = Arena(100, 60)

# create an agent and add it to the agents list
sim_agent = Agent(0, 0, BASE_TIME_STEP)
sim_agent.set_kinematic_parameters(6, 6, 6)
sim_agents = list()
sim_agents.append(sim_agent)

# setup the renderer
renderer = Renderer(sim_arena, BASE_TIME_STEP, BASE_DPI)

# start the CLI thread
cli_thread = threading.Thread(target=cli_thread_func, args=(sim_arena, sim_agents))
cli_thread.start()

while True:
    renderer.render_arena()
    for agent in sim_agents:
        agent.update()
        renderer.render_agent(agent)
    renderer.update()
