# DAPO baseline

This directory provides a complete, isolated DAPO training path. It reuses the
repository's existing verl workers and task rewards while replacing only the
driver-side trainer loop.

Implemented DAPO components:

- decoupled clipping (`clip_ratio_low=0.2`, `clip_ratio_high=0.28`)
- dynamic sampling with zero-variance group filtering
- token-level policy-gradient loss
- soft overlong reward shaping
- no actor-loss or reward KL penalty

The local reward-manager adapter also forwards truncation metadata expected by
the repository's math and code reward functions.

The trainer entry point is `python -m training.dapo.main_dapo`. It is retained
as an optional baseline; the minimal GCPO reproduction path is documented in
the repository root README.
