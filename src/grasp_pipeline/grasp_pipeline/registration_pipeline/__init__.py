"""Multi-view registration subsystem.

Owns the multi-view observation loop, NBV candidate generation,
continuous-sweep helpers, and the final view fusion / utility passes.
"""

from .multi_view import MultiViewRegistrar  # noqa: F401
