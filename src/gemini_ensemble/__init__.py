"""
gemini-ensemble: A Python library for Gemini LLM ensemble control with dynamic temperature scattering.
"""

from .client import EnsembleClient
from .reducer import BaseReducer, VotingReducer, CriticReducer

__version__ = "1.0.0"
__all__ = [
    "EnsembleClient",
    "BaseReducer",
    "VotingReducer",
    "CriticReducer",
]
