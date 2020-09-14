"""
    @file status.py
    @brief routing status definitions
    @author Graham Riches
    @details
    status definitions for routing algorithms return values.
"""

from enum import Enum


class RoutingStatus(Enum):
    SUCCESS = 0
    TARGET_BLOCKED = 1
    TARGET_RESERVED = 2
