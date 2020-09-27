"""
    @file agent_exceptions.py
    @brief contains error classes for the agent object
    @author Graham Riches
    @details
    Exceptions that can be thrown by the agent class
   
"""


class MotionError(Exception):
    """
    Exception class for agent motion errors
    """
    def __init__(self, message: str) -> None:
        self.message = message
