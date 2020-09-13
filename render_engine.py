"""
    @file render_engine.py
    @brief Renders the simulation as an animations
    @author Graham Riches
    @details
    renderer for the simulation (if enabled)
   
"""
import numpy as np
import pygame
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
        # initialize pygame as a renderer
        pygame.init()
        pygame.font.init()

        # create the rendering canvas
        self.x_size, self.y_size = arena.get_dimensions()
        self.dpi = dpi_scaling
        self._display_x = self.x_size * self.dpi
        self._display_y = self.y_size * self.dpi
        self.screen = pygame.display.set_mode((self._display_x,
                                               self._display_y))
        self.clock = pygame.time.Clock()

        # create a list of objects to render
        self.arena = arena

        # dict of color keys
        self.colors_dict = {'tile_free': (180, 180, 180), 'tile_blocked': (0, 0, 0), 'grid_lines': (255, 255, 255),
                            'agent': (165, 255, 190)}
        self.total_elements = len(self.colors_dict)

    def render_arena(self) -> None:
        """
        render the simulation arena
        :return:
        """
        for x in range(self.x_size):
            for y in range(self.y_size):
                y_pos = y*self.dpi
                x_pos = x * self.dpi
                if self.arena.get_tile_state(x, y) == TileState.FREE:
                    color = self.colors_dict['tile_free']
                else:
                    color = self.colors_dict['tile_blocked']
                # draw the tile
                rect_location = (x_pos, y_pos, self.dpi, self.dpi)
                pygame.draw.rect(self.screen, color, rect_location)
                # draw the grid rectangles
                pygame.draw.rect(self.screen, self.colors_dict['grid_lines'], rect_location, 1)

    def render_agent(self, agent: Agent) -> None:
        """
        render an agent
        :param: agent: the agent to render
        :return:
        """
        # find the agents current location range in pixels (x any y)
        x_pos = round(self.dpi * agent.location.X)
        y_pos = round(self.dpi * agent.location.Y)
        # draw the tile
        rect_location = (x_pos, y_pos, self.dpi, self.dpi)
        pygame.draw.rect(self.screen, self.colors_dict['agent'], rect_location)

    def update(self) -> None:
        """
        Update the rendering canvas
        :return: None
        """
        pygame.display.flip()
        # flush the events queue
        pygame.event.get()
