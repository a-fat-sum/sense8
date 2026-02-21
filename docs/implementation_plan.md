# sense8 — Monocular Visual-Inertial SLAM Implementation Plan (C++)

## 1) Scope and Product Direction

### Primary near-term scope
- Build a **monocular visual-inertial SLAM** system for:
  - Forward-facing camera on ground vehicles.
  - Downward-facing (possibly slightly tilted) camera on drones.
- Inputs in this phase:
  - Monocular camera stream.
  - IMU stream.
- Outputs in this phase:
  - Real-time pose estimate.
  - Local sparse map.
  - Offline trajectory quality report.
  - Playback/visualization frontend.

### Secondary mode in parallel (R&D track)
- Image-to-satellite image matching mode for global localization and map anchoring.
- Initially run as a separate module with loose coupling to VI-SLAM.

### Deferred for later phases
- Wheel odometry and GNSS fusion in core estimator.
- Multi-camera support.
- Dense reconstruction.

---

## 2) Algorithm Choices (Selected)

## 2.1 Core monocular VIO/SLAM strategy
Chosen baseline: **keyframe-based tightly coupled optimization** (VINS-style) with IMU preintegration.

Why:
- Better final accuracy and lower drift than filter-only pipelines in many real datasets.
- Handles monocular scale via IMU observability.
- Good open-source ecosystem in C++.

State estimation design:
- Sliding window nonlinear optimization over poses, velocities, IMU biases, and landmarks/inverse depth.
- IMU preintegration factor between keyframes.
- Reprojection error factors for tracked features.
- Marginalization of old states/factors to maintain real-time runtime.

### 2.2 Visual frontend
Chosen approach:
- Feature extraction/tracking: start with **ORB + KLT tracking fallback**.
- Outlier rejection: **RANSAC** with essential/fundamental model checks.
- Keyframe policy: parallax + tracked feature ratio + motion/IMU-based gating.

Optional upgrade path:
- SuperPoint + LightGlue/SuperGlue via ONNX Runtime (only after baseline is stable).

### 2.3 Initialization
- Gyro bias and gravity alignment from short IMU segment.
- Monocular visual initialization (up-to-scale) from multi-view geometry.
- Joint visual-inertial alignment to recover metric scale, gravity direction, velocity.

### 2.4 Loop closure and pose graph (Stage 3+)
- Place recognition with **DBoW2/DBoW3** (ORB vocabulary).
- Geometric verification of loop candidates.
- Pose graph optimization for global consistency.

### 2.5 Satellite matching mode (selected path)
Two-stage retrieval + geometric refinement:
1. **Global retrieval** from satellite tile database using learned cross-view descriptor.
2. **Geometric refinement** using local feature matches / homography or PnP-style constraints when applicable.

Practical implementation choice:
- Start with open cross-view models exported to ONNX for C++ inference.
- Candidate datasets and baselines listed in Section 6.

---

## 3) Open-Source Building Blocks (C++)

Core math and vision:
- **Eigen**: linear algebra.
- **Sophus**: Lie groups (SE(3), SO(3)).
- **OpenCV**: image processing, features, geometry, undistortion.

Optimization and factors:
- **Ceres Solver** (recommended for initial development) or GTSAM (alternative).
- Keep factor interfaces abstract so backend can be swapped later.

Mapping and loop closure:
- **DBoW2/DBoW3** for BoW place recognition.
- Optional **g2o** for dedicated pose graph if not using Ceres/GTSAM for graph optimization.

Infrastructure:
- **yaml-cpp** for configuration.
- **spdlog** + **fmt** for logging.
- **CLI11** for command-line tools.
- **GoogleTest** for unit tests.

Visualization frontend:
- **OpenGL + GLFW + Dear ImGui** (recommended for custom local playback UI).
- Optional **Pangolin** for rapid 3D prototyping.
- Map tiles for satellite view: HTTP tiles + local cache; use **libcurl** + lightweight raster handling.

Serialization and IO:
- Rosbag reader optional (if consuming ROS datasets).
- Native dataset adapters for non-ROS formats.

---

## 4) Proposed Repository / Module Layout

```text
sense8/
  docs/
  third_party/
  cmake/
  configs/
  datasets/
    adapters/
  core/
    common/
    camera/
    imu/
    frontend/
    backend/
    initialization/
    mapping/
    loop_closure/
    vio_pipeline/
    satloc/
  tools/
    dataset_player/
    evaluator/
    vocab_trainer/
  apps/
    sense8_vio_main/
    sense8_viewer/
  tests/
```

Key contract boundaries:
- `frontend` publishes tracked feature observations and keyframe triggers.
- `backend` consumes feature + IMU packets and outputs optimized states.
- `mapping/loop_closure` consume keyframes and publish global corrections.
- `satloc` consumes camera frames and returns geolocation hypotheses.

---

## 5) Staged Implementation Plan

## Stage 0 — Engineering Foundation (1–2 weeks)
Deliverables:
- CMake superbuild, dependency pinning, formatting/linting, CI skeleton.
- Unified timestamped sensor packet format and logging format.
- Dataset adapter interfaces (camera + IMU at minimum).

Exit criteria:
- Can replay one dataset and inspect synchronized camera/IMU streams in viewer.

## Stage 1 — Frontend + IMU Preintegration Skeleton (2–3 weeks)
Deliverables:
- Camera model and undistortion pipeline.
- ORB/KLT tracking with track lifecycle management.
- IMU integration and preintegration unit tests.
- Motion-only propagation at IMU rate.

Exit criteria:
- Stable feature tracks and IMU propagation visible in playback.
- Unit tests validate Jacobians / preintegration consistency.

## Stage 1.5 — 3D Viewer Scaffold (visualization-first, 1–2 weeks)
Deliverables:
- Add a 3D viewport in the viewer that is independent from estimator internals.
- Show ground-truth trajectory polyline from dataset adapters (where GT exists).
- Add two-view triangulated landmarks layer (debug visualization from matched features).
- Render camera frusta / pose trail for replay cursor and keyframes.
- Define trajectory channels in viewer: `GT`, `Estimated`, `Reference` (empty channels allowed).

Implementation guardrails:
- Keep this as a visualization module only; no estimator coupling.
- Use clear frame conventions in UI (`world`, `body`, `cam0`) and document transforms.
- Treat triangulated points as debug-quality (no map persistence assumptions yet).

Exit criteria:
- EuRoC replay shows GT trajectory in 3D plus optional triangulated points without flicker.
- Viewer can overlay multiple trajectory channels even if only GT is populated.

## Stage 2 — Monocular Visual-Inertial Initialization + Sliding Window VIO (4–6 weeks)
Deliverables:
- Visual initialization (up-to-scale) + IMU alignment to metric scale.
- Sliding window optimizer with reprojection + IMU factors.
- Marginalization and robust loss setup.
- Basic failure detection/reinitialization.

Exit criteria:
- Runs end-to-end on at least 2 datasets with bounded drift.
- Produce ATE/RPE reports from evaluator tool.

## Stage 3 — Mapping + Loop Closure (3–5 weeks)
Deliverables:
- Keyframe map management.
- BoW place recognition and loop candidate verification.
- Pose graph optimization + trajectory correction.

Exit criteria:
- Loop closure visibly reduces global drift on long sequences.

## Stage 4 — Productized Playback/Visualization Frontend (2–4 weeks)
Deliverables:
- Time scrub, play/pause/seek, speed control.
- Overlay tracks, keyframes, sparse map points, uncertainty indicators.
- Toggle satellite layer and estimated geo-hypotheses (for satloc mode outputs).

Exit criteria:
- Internal demo app can replay runs and compare multiple trajectories.

## Stage 5 — Satellite Matching Mode (R&D, 4–8 weeks)
Deliverables:
- Satellite tile indexing and retrieval pipeline.
- Cross-view descriptor inference path (ONNX in C++).
- Candidate ranking + geometric refinement.
- Offline benchmarking scripts and recall@K metrics.

Exit criteria:
- Reliable top-K area retrieval on selected datasets.
- Geolocation hypothesis can be visualized alongside SLAM trajectory.

## Stage 6 — Robustness & Deployment Hardening (ongoing)
Deliverables:
- Performance profiling and optimization (multi-threading, memory pools).
- Adverse-condition handling (blur, low texture, rapid dynamics).
- Better calibration tooling and health checks.

Exit criteria:
- Deterministic replay, stable long-run behavior, reproducible benchmarks.

---

## 6) Dataset Plan (No Hardware Yet)

Note: availability and exact sensor fields can evolve; verify license and modality before locking benchmarks.

## 6.1 On-road autonomous vehicle (VI focus)
1. **KITTI Raw / KITTI Odometry**
   - Why: canonical driving benchmark; supports trajectory evaluation.
   - Use: early VIO debugging, loop closure demonstrations.

2. **Oxford RobotCar**
   - Why: repeated routes, varying weather/lighting, long trajectories.
   - Use: robustness and relocalization stress tests.

3. **KAIST Urban / UrbanNav-style driving datasets**
   - Why: urban canyons and realistic vehicle motion.
   - Use: harder urban tracking and drift analysis.

## 6.2 Off-road autonomous vehicle
1. **RELLIS-3D**
   - Why: off-road terrain, vegetation, irregular motion.
   - Use: low-texture and non-planar terrain stress tests.

2. **TartanDrive / similar off-road multimodal sets (if available)**
   - Why: rough terrain and high vibration conditions.
   - Use: IMU robustness and estimator stability validation.

3. **TartanAir (synthetic but broad scenarios)**
   - Why: scalable scenario generation for failure-case testing.
   - Use: controlled ablation (blur/noise/lighting/dynamics).

## 6.3 Drone / UAV VI datasets
1. **EuRoC MAV**
   - Why: standard visual-inertial benchmark with high-quality GT.
   - Use: baseline algorithm validation and regression tests.

2. **UZH-FPV Drone Racing**
   - Why: aggressive motion and motion blur.
   - Use: dynamic robustness and failure detection tuning.

3. **Blackbird UAV Dataset**
   - Why: high-speed flight with accurate ground truth.
   - Use: high-dynamics estimator validation.

4. **TUM VI**
   - Why: photometric challenges and realistic VI data.
   - Use: tracking robustness in difficult illumination.

## 6.4 Image-to-satellite matching datasets
1. **CVUSA**
   - Ground-to-satellite retrieval benchmark.

2. **CVACT**
   - Cross-view geo-localization at larger scale.

3. **VIGOR**
   - Strong benchmark for cross-view retrieval in urban scenes.

4. **University-1652**
   - Drone-view / satellite / ground cross-view localization tasks.

---

## 7) Metrics and Progress Gates

Core VIO metrics:
- Absolute Trajectory Error (ATE).
- Relative Pose Error (RPE) over fixed windows.
- Scale drift (monocular critical metric).
- Tracking uptime (% frames successfully tracked).
- Real-time factor (processing time / sensor time).

Loop closure metrics:
- Drift reduction before vs after loop optimization.
- False loop rate / accepted loop precision.

Satellite matching metrics:
- Recall@K for tile retrieval.
- Median geolocation error (meters).
- Time-to-first-correct hypothesis.

Release gates (recommended):
- Gate A: Stage 2 passes on EuRoC + one driving dataset.
- Gate B: Stage 3 shows drift reduction on long route dataset.
- Gate C: Stage 5 reaches target Recall@K on at least two cross-view datasets.

---

## 8) Practical Development Strategy

- Implement deterministic offline replay first; real-time mode second.
- Keep one canonical config per dataset to avoid benchmark drift.
- Add automated regression suite on small sequence snippets.
- Record all run artifacts: config hash, commit hash, metrics JSON, trajectory files.

Recommended first baseline milestone:
1. EuRoC monocular+IMU VIO working.
2. KITTI/Oxford driving sequence with acceptable drift.
3. Viewer displaying 3D GT trajectory + debug triangulated landmarks + timing diagnostics.

---

## 9) Risks and Mitigations

Primary risks:
- Monocular initialization failures in low-parallax segments.
- Scale inconsistency during aggressive maneuvers.
- Off-road texture/lighting variability causing track loss.
- Cross-view (ground/drone ↔ satellite) domain gap.

Mitigations:
- Strong initialization gating and reinitialization policy.
- Robust losses + outlier rejection + IMU bias monitoring.
- Dataset diversification early (on-road/off-road/drone).
- Satellite mode kept modular until retrieval quality is stable.

---

## 10) What to Build Immediately (next 2–3 weeks)

1. Keep stabilizing CI and deterministic replay workflow.
2. Add 3D viewport to viewer with camera controls (orbit/pan/zoom) and frame axes.
3. Visualize GT trajectory from EuRoC in 3D.
4. Add two-view triangulation debug layer (from inlier matches) in viewer.
5. Introduce trajectory channels: `GT` now, `Estimated` placeholder for Stage 2.
6. Add a small validation checklist: frame alignment, scale sanity, and render stability.

This creates a strong visualization baseline now, so estimated trajectory can be dropped in later with minimal UI churn.