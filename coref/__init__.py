""" Describes a model to extract coreferential spans from a list of tokens.

  Usage example:

  model = CorefModel("config.toml", "debug")
  model.evaluate("dev")
"""

from .coref_model import CorefScorer
from .span_predictor import SpanPredictor

__all__ = [
    "CorefModel"
]
