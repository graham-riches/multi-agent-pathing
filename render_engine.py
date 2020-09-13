"""
    @file render_engine.py
    @brief Renders the simulation as an animations
    @author Graham Riches
    @details
    renderer for the simulation (if enabled)
   
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from arena import Arena
from agent import Agent
from tile import TileState


class Renderer:
    def __init__(self, arena: Arena, timestep: float, dpi_scaling: int = 40) -> None:
        """
        Create a new canvas for rendering the simulation. The base canvas is based on an Arena object,
        which controls the domain of the simulation
        :param: arena: the arena object
        :param: timestep: the animation timestep in seconds
        :param: dpi_scaling: the dpi scaling to use for one grid square
        :return:
        """
        # create the figure canvas with scaling
        self.x_size, self.y_size = arena.get_dimensions()
        self.dpi = dpi_scaling
        self.fig = plt.figure(figsize=(self.x_size, self.y_size), dpi=self.dpi)

        # create a list of objects to render
        self.arena = arena
        self.agents = list()

        # dict of color keys
        self.colors_dict = {'tile_free': 0, 'tile_blocked': 1}
        self.total_elements = len(self.colors_dict)

        # create the image grid
        self.render_grid = self.generate_grid()

        # setup animation parameters
        self._animation = None
        self.dt = int(1000*timestep)

    def close(self) -> None:
        """
        close a rendering
        :return:
        """
        self.fig.close()

    def generate_grid(self) -> np.ndarray:
        """
        generates a grid for the game
        :return: 2D array
        """
        return np.zeros((self.y_size*self.dpi, self.x_size*self.dpi))

    def render_arena(self) -> None:
        """
        render the simulation arena
        :return:
        """
        for x in range(self.x_size):
            for y in range(self.y_size):
                y_start = y*self.dpi
                y_end = y_start + self.dpi
                x_start = x * self.dpi
                x_end = x_start + self.dpi
                if self.arena.get_tile_state(x, y) == TileState.FREE:
                    self.render_grid[y_start:y_end, x_start:x_end] = self.colors_dict['tile_free']
                else:
                    self.render_grid[y_start:y_end, x_start:x_end] = self.colors_dict['tile_blocked']

    def render_agent(self, agent: Agent, agent_number: int) -> None:
        """
        render an agent
        :param: agent: the agent to render
        :param: agent_number: the number of the agent to get the colormap value
        :return:
        """
        # find the agents current location range in pixels (x any y)
        start_x_pos = int(self.dpi * agent.location.X)
        end_x_pos = start_x_pos + self.dpi
        start_y_pos = int(self.dpi * agent.location.Y)
        end_y_pos = start_y_pos + self.dpi

        # color those pixels
        self.render_grid[start_y_pos:end_y_pos, start_x_pos:end_x_pos] = agent_number

    def add_image_elements(self) -> None:
        """
        add image elements like labels etc.
        :return:
        """
        ax = plt.gca()
        # Major ticks
        ax.set_xticks(np.arange(0, self.x_size * self.dpi, self.dpi))
        ax.set_yticks(np.arange(0, self.y_size * self.dpi, self.dpi))
        # Labels for major ticks
        ax.set_xticklabels(np.arange(0, self.x_size, 1))
        ax.set_yticklabels(np.arange(0, self.y_size, 1))
        # Minor ticks
        ax.set_xticks(np.arange(-.5, self.x_size * self.dpi, self.dpi), minor=True)
        ax.set_yticks(np.arange(-.5, self.y_size * self.dpi, self.dpi), minor=True)
        # labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

    def add_agent(self, agent) -> None:
        """
        Attach a new agent to the canvas
        :return:
        """
        self.agents.append(agent)
        agent_key = 'agent_'.format(len(self.agents))
        self.colors_dict[agent_key] = self.total_elements + 1
        self.total_elements += 1

    def update(self, frame) -> None:
        """
        Update the rendering canvas
        :param: frame: required arg for matplotlib animate (not used)
        :return: None
        """
        plt.clf()
        self.render_arena()
        total_agents = len(self.agents)
        for idx, agent in enumerate(self.agents):
            agent.update()
            self.render_agent(agent, self.total_elements - total_agents + idx)
        plt.imshow(self.render_grid, cmap='tab20', vmin=0, vmax=self.total_elements)
        self.add_image_elements()

    def run(self) -> None:
        """
        function that will run an infinite loop running the animation update
        :return:
        """
        self._animation = animation.FuncAnimation(self.fig, self.update, frames=None, interval=1)
        plt.show()
