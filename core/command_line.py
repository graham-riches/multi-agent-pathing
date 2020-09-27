"""
    @file command_line.py
    @brief command line interface to the agent rendering/simulation
    @author Graham Riches
    @details
    send commands into the main rendering/simulation to make stuff happen.
    Note: the name of the method is the command line argument required to trigger the function.
"""

from core.agent import AgentCoordinates
from core.arena import Arena


class CommandLine:
    def __init__(self, arena: Arena, agents: list) -> None:
        """
        Initialize a CLI with a list of agents, an arena, and a routing manager
        """
        self.arena = arena
        self.agents = agents

    def help(self, args: list) -> bool:
        """
        print out the command line help by dumping each commands doc_string
        :param: args: the command line arguments for the help function
        """
        command_id = args[0]
        retval = False
        if len(args) == 1:
            start_str = 'Menu available commands: \n\t\t'
            methods = [method for method in self.__dir__() if 'help' not in method]
            # filter out private methods
            commands = [method for method in methods if '__' not in method]
            help_str = ''.join('{} '.format(cmd) for cmd in commands)
            print('{} {}'.format(start_str, help_str))
            retval = True
        else:
            # print other specific function help
            command = [method for method in self.__dir__() if method == command_id]
            if command:
                docstring = getattr(self, command[0])
                print(docstring.__doc__)
                retval = True
        return retval

    def move_agent(self, args: list) -> bool:
        """
        Send a command to start an agent move
            Usage: agent_move [agent_id] [direction] [distance]
            [agent_id] - the identifier of the agent in the simulation
            [direction] - movement direction X, or Y
            [distance] - number of squares to move
        """
        if len(args) < 3:
            self.help(['move_agent'])
            return False
        id = int(args[0])
        direction = AgentCoordinates.X if args[1] == 'X' else AgentCoordinates.Y
        distance = int(args[2])
        target_agent = self.agents[id]
        target_agent.start_move(direction, distance)
        return True

    def blockage(self, args: list) -> bool:
        """
        Set a blockage in the game arena
            Usage: blockage [set/clear] [x] [y]
            [type] - 'set' or 'clear'
        """
        if len(args) < 3:
            self.help(['blockage'])
            return False
        x = int(args[1])
        y = int(args[2])
        if args[0] == 'set':
            self.arena.set_blockage([x], [y])
        elif args[0] == 'clear':
            self.arena.clear_blockage([x], [y])
        else:
            return False
        return True

    def parse_command(self, command: str) -> bool:
        """
        parse a command
        """
        args = command.split(' ')
        cmd_str = '{}'.format(args[0])
        arg_str = ''.join('{} '.format(x) for x in args[1:])
        args = arg_str.split(' ')

        # find the arg in the class properties and print out the docstring
        method_str = [method for method in self.__dir__() if method == cmd_str]

        # call the command with the arguments
        if method_str:
            retval = getattr(self, method_str[0])(args)
            print('Command Received: {}'.format(method_str[0]))
            return retval
        else:
            return False

    def get_input(self) -> bool:
        """
        Get user input and parse it to execute commands
        """
        user_input = input()
        return self.parse_command(user_input)
