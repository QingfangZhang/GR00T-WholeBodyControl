"""Non-invasive reference-motion overlay for the SONIC MuJoCo simulator."""

from .ghost_overlay import (
    GhostOverlayConfig,
    ReferenceGhostError,
    ReferenceGhostOverlay,
    RootMode,
    TargetPose,
)

__all__ = [
    "GhostOverlayConfig",
    "ReferenceGhostError",
    "ReferenceGhostOverlay",
    "RootMode",
    "TargetPose",
]
