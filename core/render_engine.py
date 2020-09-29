"""
    @file render_engine.py
    @brief Renders the simulation as an animations
    @author Graham Riches
    @details
    renderer for the simulation (if enabled)
   
"""
import pygame
from core.agent_colors import COLORS
from core.arena import Arena
from core.tile import TileState
from routing.routing_algorithm import MultiAgentAlgorithm


# pygame buttons
LEFT = 1
RIGHT = 3


class Renderer:
    def __init__(self, arena: Arena, agents: list, routing_manager: MultiAgentAlgorithm,
                 timestep: float, dpi_scaling: int = 40) -> None:
        """
        Create a new canvas for rendering the simulation. The base canvas is based on an Arena object,
        which controls the domain of the simulation
        :param: arena: the arena object
        :param: agents: list of agents
        :param: routing_manager: routing manager object
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

        self.arena = arena
        self.agents = agents
        self.routing_manager = routing_manager
        self.agent_selected = None

        # dict of color keys
        self.colors_dict = {'tile_free': (180, 180, 180), 'tile_blocked': (0, 0, 0), 'tile_reserved': (60, 60, 60),
                            'tile_target': (200, 135, 135), 'grid_lines': (255, 255, 255), 'agent': COLORS,
                            'agent_selected': (245, 100, 90), 'agent_border': (0, 0, 0)}
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
                elif self.arena.get_tile_state(x, y) == TileState.BLOCKED:
                    color = self.colors_dict['tile_blocked']
                elif self.arena.get_tile_state(x, y) == TileState.AGENT_TARGET:
                    color = self.colors_dict['tile_target']
                else:
                    color = self.colors_dict['tile_reserved']
                # draw the tile
                rect_location = (x_pos, y_pos, self.dpi, self.dpi)
                pygame.draw.rect(self.screen, color, rect_location)
                # draw the grid rectangles
                pygame.draw.rect(self.screen, self.colors_dict['grid_lines'], rect_location, 1)

    def render_agent(self, agent_id: int) -> None:
        """
        render an agent
        :param: agent_id: the id of the agent to render
        :return:
        """
        agent = self.agents[agent_id]
        # find the agents current location range in pixels (x any y)
        x_pos = round(self.dpi * agent.location.X)
        y_pos = round(self.dpi * agent.location.Y)
        # draw the tile
        rect_location = (x_pos, y_pos, self.dpi, self.dpi)
        color = self.colors_dict['agent_selected'] if agent_id == self.agent_selected else self.colors_dict['agent'][agent_id]
        pygame.draw.rect(self.screen, color, rect_location)
        pygame.draw.rect(self.screen, self.colors_dict['agent_border'], rect_location, 1)

    def blockage_on_click(self, block: bool, x_index: int, y_index: int) -> None:
        """
        add or remove a blockage on a click event
        :param block: boolean true/false
        :param x_index: x location
        :param y_index: y location
        :return: None
        """
        if block:
            self.arena.set_blockage([x_index], [y_index])
        else:
            self.arena.clear_blockage([x_index], [y_index])

    def handle_click_event(self, button, x_position: int, y_position: int) -> None:
        """
        handle a click event on the game grid
        :param button: pygame button type
        :param x_position: x location in pixels
        :param y_position: y location in pixels
        """
        # TODO some of this function is a bit hacky ¯\_(ツ)_/¯
        # convert the pixel locations to grid indices
        x_ind = int(x_position/self.dpi)
        y_ind = int(y_position/self.dpi)

        # check for agents at the clicked location
        found_agent = False
        for idx, agent in enumerate(self.agents):
            if agent.location.X == x_ind and agent.location.Y == y_ind:
                found_agent = True
                self.agent_selected = idx

        if not found_agent:
            if self.agent_selected is not None:
                self.routing_manager.route(self.agent_selected, (x_ind, y_ind))
                self.agent_selected = None
            else:
                if button == LEFT:
                    self.blockage_on_click(True, x_ind, y_ind)
                else:
                    self.blockage_on_click(False, x_ind, y_ind)

    def update(self) -> None:
        """
        Update the rendering canvas
        :return: None
        """
        pygame.display.flip()
        # flush the events queue
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                self.handle_click_event(event.button, pos[0], pos[1])
