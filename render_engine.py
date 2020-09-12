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
from matplotlib.colors import ListedColormap
from abc import ABC, abstractmethod
from arena import Arena
from tile import TileState


TILE_FREE_COLOR = (0.8, 0.8, 0.8)
TILE_BLOCKED_COLOR = (0, 0, 0)
TILE_COLORS = [TILE_BLOCKED_COLOR, TILE_FREE_COLOR]
TILE_CMAP = ListedColormap(TILE_COLORS, 'Tile', len(TILE_COLORS))


class Renderer(ABC):
    """
    Base class that individual rendering objects should be based on
    """
    @abstractmethod
    def render(self) -> None:
        pass


class ArenaRenderer(Renderer):
    def __init__(self, arena: Arena) -> None:
        """
        Create an arena renderer
        :param arena: the arena object to render
        :return: None
        """
        self._arena = arena
        self._x_size, self._y_size = self._arena.get_dimensions()

    def render(self) -> None:
        """
        Generate a rendering of an arena.
        :return:
        """
        # convert the arena into numerical values
        plot_grid = np.ndarray((self._y_size, self._x_size))
        for x in range(self._x_size):
            for y in range(self._y_size):
                if self._arena.get_tile_state(x, y) == TileState.FREE:
                    plot_grid[y][x] = 1
                else:
                    plot_grid[y][x] = 0
        plt.imshow(plot_grid, cmap=TILE_CMAP, vmin=0, vmax=1)
        ax = plt.gca()

        # Major ticks
        ax.set_xticks(np.arange(0, self._x_size, 1))
        ax.set_yticks(np.arange(0, self._y_size, 1))

        # Labels for major ticks
        ax.set_xticklabels(np.arange(0, self._x_size, 1))
        ax.set_yticklabels(np.arange(0, self._y_size, 1))

        # Minor ticks
        ax.set_xticks(np.arange(-.5, self._x_size, 1), minor=True)
        ax.set_yticks(np.arange(-.5, self._y_size, 1), minor=True)

        # labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)


class Canvas:
    def __init__(self, arena: Arena, timestep: float) -> None:
        """
        Create a new canvas for rendering the simulation. The base canvas is based on an Arena object,
        which controls the domain of the simulation
        :param: arena: the arena object
        :param: timestep: the animation timestep in seconds
        :return:
        """
        x_size, y_size = arena.get_dimensions()
        self.fig = plt.figure(figsize=(y_size, x_size))

        # create a list of objects to render
        self.arena_renderer = ArenaRenderer(arena)
        self._render_objects = [self.arena_renderer]

        # create an animation
        self._animation = None
        self.dt = int(1000*timestep)

    def attach_renderer(self, renderer) ->None:
        """
        Attach a new renderer to the canvas
        :param renderer: the renderer object (inherits from Renderer and thus has .render() method)
        :return:
        """
        self._render_objects.append(renderer)

    def update(self, frame) -> None:
        """
        Update the rendering canvas
        :param: frame: required arg for matplotlib animate (not used)
        :return: None
        """
        for render_task in self._render_objects:
            render_task.render()

    def run(self, test_mode=False) -> None:
        """
        function that will run an infinite loop running the animation update
        :param: test_mode: flag to enable a single frame test mode
        :return:
        """
        self._animation = animation.FuncAnimation(self.fig, self.update, frames=None, interval=self.dt)
        plt.show()
