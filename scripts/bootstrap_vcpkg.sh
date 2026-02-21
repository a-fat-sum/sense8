#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vcpkg_root="${repo_root}/.vcpkg"

if [[ ! -d "${vcpkg_root}" ]]; then
  echo "Cloning vcpkg into ${vcpkg_root}"
  git clone https://github.com/microsoft/vcpkg.git "${vcpkg_root}"
fi

echo "Bootstrapping vcpkg..."
"${vcpkg_root}/bootstrap-vcpkg.sh"

echo
cat <<EOF
Set VCPKG_ROOT for this shell:
  export VCPKG_ROOT="${vcpkg_root}"
Then configure with:
  cmake --preset debug
EOF
