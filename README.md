# sense8

Monocular visual-inertial SLAM system in C++ with optional satellite matching mode.

## Command line parsing

The project uses **CLI11** as the C++ argument parser (`argparse`) library.

## Dependency management

This repository uses **vcpkg manifest mode** (`vcpkg.json`) so dependencies are defined in-repo and restored automatically for local builds and CI.

### Why vcpkg here
- Single dependency manifest tracked in git.
- Cross-platform support for Windows and Ubuntu.
- Reproducible CI by pinning `builtin-baseline`.
- No manual build/install of individual third-party libraries.

## Quick start (Windows PowerShell)

```powershell
./scripts/bootstrap_vcpkg.ps1
$env:VCPKG_ROOT = "$PWD/.vcpkg"
cmake --preset debug
cmake --build --preset debug --parallel
ctest --preset ci --output-on-failure
```

## Quick start (Ubuntu)

```bash
./scripts/bootstrap_vcpkg.sh
export VCPKG_ROOT="$PWD/.vcpkg"
cmake --preset debug
cmake --build --preset debug --parallel
ctest --preset ci --output-on-failure
```

## CI

GitHub Actions workflow at `.github/workflows/ci.yml` builds and tests on:
- `ubuntu-latest`
- `windows-latest`

## Optional Dear ImGui viewer

Viewer target `sense8_viewer` is available behind `SENSE8_BUILD_VIEWER`.

```powershell
cmake --preset debug -DSENSE8_BUILD_VIEWER=ON
cmake --build --preset debug --parallel
./build/debug/apps/sense8_viewer/Debug/sense8_viewer.exe --dataset E:/Repos/sense8/datasets/euroc/MH_01_easy
```

```bash
cmake --preset debug -DSENSE8_BUILD_VIEWER=ON
cmake --build --preset debug --parallel
./build/debug/apps/sense8_viewer/sense8_viewer --dataset ./datasets/euroc/MH_01_easy
```

## EuRoC dataset location

Full EuRoC datasets are **not downloaded automatically**.

Place extracted sequences under:

- `datasets/euroc/<sequence_name>/mav0/...`

Example:

- `datasets/euroc/MH_01_easy/mav0/imu0/data.csv`
- `datasets/euroc/MH_01_easy/mav0/cam0/data.csv`

## Current skeleton targets
- `sense8_common` (core shared library)
- `sense8_vio_main` (CLI app entrypoint)
- `sense8_common_tests` (GoogleTest)
