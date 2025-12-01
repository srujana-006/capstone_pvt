# services/__init__.py
"""
Services module: Business logic layer for the simulation.

Contains service classes that orchestrate domain logic across entities:
  - DispatcherService: Gridwise dispatch of EVs to incidents (Algorithm 2)
  - RepositioningService: EV repositioning offers and acceptance (Algorithm 1)
  - NavigationService: Hospital selection and routing
"""

from .dispatcher import DispatcherService
from .repositioning import RepositioningService
from .navigation import NavigationService

__all__ = [
    "DispatcherService",
    "RepositioningService",
    "NavigationService",
]
