"""Explicit hand-torque profiles for controller-replacement rollouts.

The recorded task XML and the released SONIC simulator do not use the same
Dex3 effort limits for six of the seven joints.  Keeping that choice in a
named profile makes the formal 0.7 Nm limit auditable while retaining the
recording XML's 1.4 Nm range as an explicit sensitivity condition.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np


FloatArray = np.ndarray
SONIC_RELEASE_HAND_TORQUE_LIMIT_NM = np.asarray(
    [2.45, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7], dtype=np.float64
)


class HandTorqueProfile(str, Enum):
    """Named software limit applied before the staged XML actuator range."""

    SONIC_RELEASE = "sonic_release"
    STAGED_XML = "staged_xml"

    @classmethod
    def parse(cls, value: "HandTorqueProfile | str") -> "HandTorqueProfile":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "_")
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"unsupported hand torque profile {value!r}; expected one of {choices}"
            ) from exc

    @property
    def software_torque_limit_nm(self) -> FloatArray | None:
        """Return the symmetric profile limit, or ``None`` for XML-only."""

        if self is HandTorqueProfile.SONIC_RELEASE:
            return SONIC_RELEASE_HAND_TORQUE_LIMIT_NM.copy()
        return None

    def clip(self, torque: Any) -> tuple[FloatArray, FloatArray]:
        """Apply this profile and return ``(torque, saturation_mask)``."""

        value = np.asarray(torque, dtype=np.float64).reshape(-1)
        if value.shape != (7,):
            raise ValueError(f"hand torque has shape {value.shape}; expected (7,)")
        if not np.all(np.isfinite(value)):
            raise ValueError("hand torque contains NaN or infinity")
        limit = self.software_torque_limit_nm
        if limit is None:
            return value.copy(), np.zeros(7, dtype=np.bool_)
        clipped = np.clip(value, -limit, limit)
        return clipped, clipped != value

    def metadata(self) -> dict[str, object]:
        limit = self.software_torque_limit_nm
        return {
            "profile": self.value,
            "software_torque_limit_nm": (
                None if limit is None else limit.tolist()
            ),
            "software_limit_semantics": (
                "released SONIC motor_effort_limit_list"
                if self is HandTorqueProfile.SONIC_RELEASE
                else "none; staged MuJoCo actuator ctrlrange only"
            ),
        }


__all__ = [
    "HandTorqueProfile",
    "SONIC_RELEASE_HAND_TORQUE_LIMIT_NM",
]
