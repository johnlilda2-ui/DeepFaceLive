# PoseCap-derived retargeting code

Upstream: https://github.com/CorridorTech/PoseCap
Revision: 626701bd6315727e791cdc9c1353830b9492d6ee
License: Apache-2.0 (see LICENSE in this directory).

scripts/avatar_body_retarget.py adapts core/src/posecap_core/landmark_pose.py and
rotation.py. The neutral test fixture in scripts/test_avatar_body_retarget.py
derives from tests/core/test_landmark_pose.py.

Changes: standalone worker packet output, camera coordinate alignment,
confidence gating, temporal pole handling, leg-direction priority, and
three-point foot frames. No PoseCap application, installer, Blender component,
model binary or PEAR runtime is included. This does not change separate
licenses applicable to the existing LHM++/SMPL-X runtime assets.
