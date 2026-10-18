"""Transformer with the actions zeroed. Its gap to `transformer` is the learned action effect."""
from wmpoc.models.world_models.transformer import TransformerWorldModel
from wmpoc.registry import WORLD_MODELS


@WORLD_MODELS.register("no_action")
class NoActionWorldModel(TransformerWorldModel):
    zero_actions = True
