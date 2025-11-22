import os
import sys

from .ghostbusters_env import GhostbustersPyMARLEnv


REGISTRY = {} if "REGISTRY" not in globals() else REGISTRY
REGISTRY["ghostbusters"] = lambda **kwargs: GhostbustersPyMARLEnv(**kwargs)


def register_smac():
    """Register SMAC environment (lazy import to avoid pysc2 conflicts)"""
    from .smac_wrapper import SMACWrapper
    REGISTRY["sc2"] = lambda **kwargs: SMACWrapper(**kwargs)


def register_smacv2():
    """Register SMACv2 environment (lazy import to avoid pysc2 conflicts)"""
    from .smacv2_wrapper import SMACv2Wrapper
    REGISTRY["sc2v2"] = lambda **kwargs: SMACv2Wrapper(**kwargs)