#!/usr/bin/env python3
"""Run the official SONIC MuJoCo simulator with an in-window reference ghost.

Only this standalone launcher and its sibling modules are custom. The official
``gear_sonic`` simulator files remain untouched and are imported as-is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Literal


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tyro  # noqa: E402

from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator  # noqa: E402
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig  # noqa: E402
from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory  # noqa: E402
from sim_reference_overlay.ghost_overlay import (  # noqa: E402
    GhostOverlayConfig,
    ReferenceGhostError,
    ReferenceGhostOverlay,
)


@dataclass
class ReferenceGhostSimConfig(SimLoopConfig):
    """Official simulation options plus visual-only reference ghost settings."""

    ghost_root_mode: Literal["actual", "reference"] = "reference"
    """Use actual XYZ, or a first-frame-aligned reference XYZ trajectory."""

    ghost_alpha: float = 0.30
    """Reference ghost opacity in (0, 1]."""

    ghost_zmq_host: str = "localhost"
    """Host running the C++ deployment ZMQ output."""

    ghost_zmq_port: int = 5557
    """C++ deployment output port."""

    ghost_zmq_topic: str = "g1_debug"
    """C++ deployment output topic."""

    ghost_stale_timeout: float = 1.0
    """Hide the ghost after this many seconds without a valid target."""

    ghost_visual_geom_group: int = 1
    """MuJoCo visual geom group copied into the user scene."""


class ReferenceGhostSimulator(BaseSimulator):
    """Wrap ``BaseSimulator`` at runtime without editing its source file."""

    def __init__(self, *, ghost_config: GhostOverlayConfig, **kwargs) -> None:
        self.reference_ghost: ReferenceGhostOverlay | None = None
        self._viewer_update_without_ghost = None
        super().__init__(**kwargs)
        try:
            self.reference_ghost = ReferenceGhostOverlay(
                model=self.sim_env.mj_model,
                actual_data=self.sim_env.mj_data,
                viewer=self.sim_env.viewer,
                config=ghost_config,
            )
        except Exception:
            super().close()
            raise

        # BaseSimulator's loop already calls sim_env.update_viewer at VIEWER_DT.
        # Replace only this instance's bound method, leaving the official class
        # and source untouched. Physics stepping and DDS behavior are unchanged.
        self._viewer_update_without_ghost = self.sim_env.update_viewer
        self.sim_env.update_viewer = self._update_viewer_with_reference_ghost

    def _update_viewer_with_reference_ghost(self) -> None:
        if self.reference_ghost is not None:
            try:
                self.reference_ghost.update()
            except Exception as exc:
                # The overlay is diagnostic only. A render/FK/subscriber bug
                # must never stop SONIC control or MuJoCo physics.
                print(
                    "[reference-ghost] disabling overlay after runtime error; "
                    f"physics will continue: {exc}",
                    file=sys.stderr,
                )
                failed_overlay, self.reference_ghost = self.reference_ghost, None
                try:
                    failed_overlay.close()
                except Exception as close_exc:
                    print(
                        "[reference-ghost] overlay cleanup also failed; "
                        f"physics will still continue: {close_exc}",
                        file=sys.stderr,
                    )
        if self._viewer_update_without_ghost is not None:
            self._viewer_update_without_ghost()

    def close(self) -> None:
        overlay, self.reference_ghost = self.reference_ghost, None
        if overlay is not None:
            overlay.close()
        super().close()


def main(config: ReferenceGhostSimConfig) -> int:
    if config.simulator != "mujoco":
        raise ReferenceGhostError(
            "this overlay supports only --simulator mujoco, got "
            f"{config.simulator!r}"
        )
    if not config.enable_onscreen:
        raise ReferenceGhostError(
            "the reference overlay requires onscreen rendering; do not pass "
            "--no-enable-onscreen"
        )
    if config.enable_image_publish and not config.enable_offscreen:
        raise ReferenceGhostError(
            "--enable-image-publish requires --enable-offscreen, matching the "
            "official simulator"
        )

    ghost_config = GhostOverlayConfig(
        root_mode=config.ghost_root_mode,
        alpha=config.ghost_alpha,
        host=config.ghost_zmq_host,
        port=config.ghost_zmq_port,
        topic=config.ghost_zmq_topic,
        stale_timeout_s=config.ghost_stale_timeout,
        visual_geom_group=config.ghost_visual_geom_group,
    )
    wbc_config = config.load_wbc_yaml()
    wbc_config["ENV_NAME"] = config.env_name

    simulator = ReferenceGhostSimulator(
        ghost_config=ghost_config,
        config=wbc_config,
        env_name=config.env_name,
        onscreen=wbc_config.get("ENABLE_ONSCREEN", True),
        offscreen=wbc_config.get("ENABLE_OFFSCREEN", False),
        enable_image_publish=config.enable_image_publish,
    )
    print(
        "[reference-ghost] solid robot = SONIC/MuJoCo result; "
        "transparent green robot = reference target"
    )
    SimulatorFactory.start_simulator(
        simulator,
        as_thread=False,
        enable_image_publish=config.enable_image_publish,
        mp_start_method=config.mp_start_method,
        camera_port=config.camera_port,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(tyro.cli(ReferenceGhostSimConfig)))
    except ReferenceGhostError as exc:
        print(f"Reference ghost setup failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
