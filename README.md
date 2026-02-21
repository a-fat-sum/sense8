# sense8

Monocular visual-inertial SLAM system in C++ with optional satellite matching mode.

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

## Current skeleton targets
- `sense8_common` (core shared library)
- `sense8_vio_main` (CLI app entrypoint)
- `sense8_common_tests` (GoogleTest)
