# Third-party provenance

The Teleopit observation layout, ONNX history behavior, action transform, G1
joint order, and numerical controller parameters used here are derived from:

- Project: <https://github.com/BotRunner64/Teleopit>
- Version: `v0.5.0`
- Commit: `f9263865c581802ad531854b8e547e2403a945f3`
- Copyright: 2026 BotRunner64
- License: Apache License 2.0

A copy of the upstream license is included as `LICENSE.Teleopit`.

The rollout adapter is a separate integration for the recorded MuJoCo task
scenes in GR00T-WholeBodyControl. It is not an upstream Teleopit entry point.

The ignored runtime assets are pinned independently from the Hugging Face
repository `12e21/Teleopit-models`, tag `v0.5.0`, commit
`94cf996444fea6894b87c28e86606cd4c2f1408f`; the downloader verifies the exact
model, archive, and robot-XML SHA-256 values before use.
