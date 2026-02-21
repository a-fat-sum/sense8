# AGENTS.md — sense8 agent instructions

These instructions apply to the entire repository.

## Mission
- Build `sense8` as a C++-first monocular visual-inertial SLAM system with a local playback/visualization frontend.
- Prioritize a stable, testable VIO baseline before adding optional modes (loop closure refinements, satellite matching, wheel odometry, GNSS).

## Ground rules
- Keep changes focused and minimal; avoid unrelated refactors.
- Prefer deterministic offline replay correctness over premature real-time optimization.
- Do not introduce new runtime dependencies unless they are justified and documented.
- Keep APIs and module boundaries explicit; avoid tight cross-module coupling.

## Language, tooling, and style
- Primary language: modern C++ (C++20 unless project config requires otherwise).
- Build system: CMake.
- Use these libraries where appropriate: Eigen, OpenCV, Sophus, Ceres (or GTSAM if selected), yaml-cpp, spdlog/fmt.
- Follow existing formatting/lint configuration when present; if absent, keep style consistent within each edited file.
- Avoid one-letter variable names except for standard math loop indices.
- Avoid adding inline comments unless needed to clarify non-obvious math or invariants.

## Architecture expectations
- Preserve and reinforce clear boundaries between:
	- `frontend` (feature extraction/tracking, outlier rejection, keyframe triggers)
	- `backend` (state estimation, factors, optimization, marginalization)
	- `mapping`/`loop_closure` (global consistency)
	- `satloc` (image-to-satellite matching path)
	- `tools`/`apps` (dataset playback, evaluator, UI)
- New features must connect through interfaces/contracts, not ad-hoc shared state.

## SLAM/VIO implementation policy
- Start with robust baseline methods before advanced ML alternatives.
	- Example baseline preference: ORB/KLT tracking + RANSAC + IMU preintegration + sliding-window optimization.
- Add failure detection and reinitialization paths early.
- Keep estimator assumptions explicit (sensor frame conventions, gravity direction, timestamp units, bias models).
- Any change affecting estimation math should include at least one targeted validation (unit/integration/benchmark check).

## Dataset and evaluation policy
- Since hardware is not available, prioritize progress on public datasets.
- Any new algorithmic change should be evaluated on at least one representative dataset sequence.
- Track and report core metrics when possible:
	- ATE, RPE, scale drift, tracking uptime, runtime factor.
- Keep dataset-specific hacks isolated behind dataset adapters or config files.

## Configuration and reproducibility
- Parameters should live in versioned config files, not hard-coded constants.
- Prefer explicit units in parameter names (for example `_ms`, `_hz`, `_rad`).
- For experiments, record enough metadata to reproduce runs (config, commit, dataset sequence).

## Visualization frontend policy
- Local viewer should support deterministic playback controls (play/pause/seek/speed).
- Keep visualization optional and non-blocking for headless evaluation tools.
- Do not couple core estimator logic to UI rendering code.

## Testing and validation expectations
- Add or update tests close to changed code when practical.
- Prefer fast, focused tests first (math/factor/preintegration), then broader integration checks.
- If tests are unavailable, provide a concrete manual validation recipe in the PR/summary.

## Performance and safety constraints
- Do not optimize blindly; profile before large performance refactors.
- Avoid unnecessary copies in hot paths; prefer move semantics and preallocation where helpful.
- Be careful with threading: document ownership, synchronization, and data lifetime.

## Dependency and licensing hygiene
- Favor permissive open-source dependencies.
- Document new dependencies (why needed, license, where used).
- Avoid adding large frameworks when a small existing dependency solves the need.

## Agent behavior expectations
- Before major edits, inspect nearby code for conventions and existing patterns.
- After edits, run the most relevant available checks/tests for changed areas.
- Summaries should state: what changed, why, how validated, and any known risks.
- If blocked by ambiguity, choose the simplest valid interpretation and proceed; note assumptions clearly.

