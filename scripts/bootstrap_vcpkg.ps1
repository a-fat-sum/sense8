$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$vcpkgRoot = Join-Path $repoRoot '.vcpkg'

if (-not (Test-Path $vcpkgRoot)) {
    Write-Host "Cloning vcpkg into $vcpkgRoot"
    git clone https://github.com/microsoft/vcpkg.git $vcpkgRoot
}

Write-Host 'Bootstrapping vcpkg...'
& (Join-Path $vcpkgRoot 'bootstrap-vcpkg.bat')

Write-Host ''
Write-Host "Set VCPKG_ROOT for this shell:"
Write-Host "  `$env:VCPKG_ROOT = '$vcpkgRoot'"
Write-Host "Then configure with:"
Write-Host "  cmake --preset debug"
