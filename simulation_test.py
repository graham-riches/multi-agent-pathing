"""
    @file simulation_test.py
    @brief quick test sim to try out the agent/arena and rendering
    @author Graham Riches
    @details
   
"""
import threading
from render_engine import Renderer
from command_line import CommandLine
from routing.routing_manager import *
from agent import *
from arena import Arena

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
sim_agent = Agent(15, 20, BASE_TIME_STEP)
sim_agent_1 = Agent(5, 5, BASE_TIME_STEP)
sim_agent_1.set_kinematic_parameters(4, 4, 4)
sim_agent.set_kinematic_parameters(6, 6, 6)
sim_agents = list()
sim_agents.append(sim_agent)
sim_agents.append(sim_agent_1)

# setup the renderer
renderer = Renderer(sim_arena, BASE_TIME_STEP, BASE_DPI)

# start the CLI thread
cli_thread = threading.Thread(target=cli_thread_func, args=(sim_arena, sim_agents))
cli_thread.start()

# create a routing manager and use it to queue some tasks for our agent
routing_manager = RoutingManager(sim_arena, sim_agents)
task_1 = AgentTask(AgentTasks.MOVE, [AgentCoordinates.X, 10])
task_2 = AgentTask(AgentTasks.MOVE, [AgentCoordinates.Y, 10])
task_3 = AgentTask(AgentTasks.MOVE, [AgentCoordinates.X, -10])
task_4 = AgentTask(AgentTasks.MOVE, [AgentCoordinates.Y, -10])
loop_tasks = [task_1, task_2, task_3, task_4]
routing_manager.add_agent_task(0, task_1)
routing_manager.add_agent_task(1, task_1)
for task in loop_tasks:
    routing_manager.add_agent_task(0, task)
    routing_manager.add_agent_task(1, task)

task_ids = [0, 0]
while True:
    renderer.render_arena()
    for agent_id, agent in enumerate(sim_agents):
        state = agent.update()
        if state == AgentState.IDLE:
            state = agent.update()
            routing_manager.signal_agent_event(agent_id, AgentEvent.TASK_COMPLETED)
            routing_manager.add_agent_task(agent_id, loop_tasks[task_ids[agent_id]])
            task_ids[agent_id] += 1
            if task_ids[agent_id] == len(loop_tasks):
                task_ids[agent_id] = 0
        renderer.render_agent(agent)
    renderer.update()
