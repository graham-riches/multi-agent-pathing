"""
    @file simulation.py
    @brief quick test sim to try out the agent/arena and rendering
    @author Graham Riches
    @details
   
"""
import threading
from render_engine import Renderer
from command_line import CommandLine
from routing.routing_manager import *
from routing.a_star import AStar
from agent import *
from arena import Arena

BASE_TIME_STEP = 0.05
BASE_DPI = 40


# CLI thread
def cli_thread_func(arena: Arena, agents: list, manager: RoutingManager) -> None:
    cli = CommandLine(arena, agents, manager)
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

# create a routing manager and use it to queue some tasks for our agent
routing_manager = RoutingManager(sim_arena, sim_agents, algorithm)

# setup the renderer
renderer = Renderer(sim_arena, sim_agents, routing_manager, BASE_TIME_STEP, BASE_DPI)

# start the CLI thread
cli_thread = threading.Thread(target=cli_thread_func, args=(sim_arena, sim_agents, routing_manager))
cli_thread.start()


while True:
    renderer.render_arena()
    for agent_id, agent in enumerate(sim_agents):
        state = agent.update()
        if state == AgentState.IDLE:
            routing_manager.signal_agent_event(agent_id, AgentEvent.TASK_COMPLETED)
        renderer.render_agent(agent_id)
    renderer.update()
