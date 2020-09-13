"""
    @file simulation_test.py
    @brief quick test sim to try out the agent/arena and rendering
    @author Graham Riches
    @details
   
"""
from render_engine import Renderer
from agent import Agent, AgentCoordinates, AgentState
from arena import Arena

BASE_TIME_STEP = 0.05
BASE_DPI = 80

# setup a quick arena
arena = Arena(10, 10)

# setup the renderer
renderer = Renderer(arena, BASE_TIME_STEP, BASE_DPI)

# create an agent and add it to the renderer
agent = Agent(0, 0, BASE_TIME_STEP)
renderer.add_agent(agent)

agent.start_move(AgentCoordinates.X, 4)
renderer.run()
