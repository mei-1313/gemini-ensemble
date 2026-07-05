"""
gemini-ensemble: A Python library for Gemini LLM ensemble control with dynamic temperature scattering.
"""

from .client import EnsembleClient, generate_ensemble
from .reducer import BaseReducer, VotingReducer, CriticReducer, reduce_voting, reduce_critic

__version__ = "1.0.0"
__all__ = [
    "EnsembleClient",
    "BaseReducer",
    "VotingReducer",
    "CriticReducer",
    "generate_ensemble",
    "reduce_voting",
    "reduce_critic",
]
