$ErrorActionPreference = 'Stop'

function Invoke-Step {
  param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Args
  )

  & $Exe @Args
  if ($LASTEXITCODE -ne 0) {
    throw "$Exe failed with exit code $LASTEXITCODE"
  }
}

if (Test-Path Env:RUSTFLAGS) {
  Remove-Item Env:RUSTFLAGS
}
if (Test-Path Env:CARGO_ENCODED_RUSTFLAGS) {
  Remove-Item Env:CARGO_ENCODED_RUSTFLAGS
}
if (Test-Path Env:RUSTDOCFLAGS) {
  Remove-Item Env:RUSTDOCFLAGS
}

Write-Host 'Building wasm (web feature set)...'
Invoke-Step cargo build --target wasm32-unknown-unknown --release --no-default-features --features web

$wasmPath = Join-Path 'target/wasm32-unknown-unknown/release' 'bevy_gaussian_splatting.wasm'
if (-not (Test-Path $wasmPath)) {
  throw "wasm output not found at $wasmPath"
}

if (-not (Get-Command wasm-bindgen -ErrorAction SilentlyContinue)) {
  throw 'wasm-bindgen not found on PATH'
}

$wasmBindgenLockVersion = $null
$sawWasmBindgen = $false
foreach ($line in Get-Content 'Cargo.lock') {
  if ($line -eq 'name = "wasm-bindgen"') {
    $sawWasmBindgen = $true
    continue
  }

  if ($sawWasmBindgen -and $line -match '^version = "([^"]+)"$') {
    $wasmBindgenLockVersion = $Matches[1]
    break
  }
}

if (-not $wasmBindgenLockVersion) {
  throw 'failed to resolve wasm-bindgen version from Cargo.lock'
}

$wasmBindgenCliVersion = (& wasm-bindgen --version).Split(' ', [System.StringSplitOptions]::RemoveEmptyEntries)[1]
if ($wasmBindgenCliVersion -ne $wasmBindgenLockVersion) {
  throw "wasm-bindgen CLI version $wasmBindgenCliVersion does not match Cargo.lock wasm-bindgen $wasmBindgenLockVersion. Install the matching CLI with: cargo install wasm-bindgen-cli --version $wasmBindgenLockVersion --locked --force"
}

Write-Host 'Generating wasm bindings...'
Invoke-Step wasm-bindgen --out-dir ./www/out --target web $wasmPath

Write-Host 'Rendering example thumbnails from manifest...'
$env:RENDER_EXAMPLE_THUMBNAILS = '1'
$env:THUMBNAIL_SORT_MODE = 'std'
try {
  Invoke-Step cargo test --test headless_examples render_example_thumbnails -- --nocapture
} finally {
  Remove-Item Env:THUMBNAIL_SORT_MODE -ErrorAction SilentlyContinue
  Remove-Item Env:RENDER_EXAMPLE_THUMBNAILS -ErrorAction SilentlyContinue
}

if ($env:THUMBNAIL_SCENE_CACHE_CLEANUP -eq '1') {
  $sceneCache = Join-Path 'assets' '.thumbnail_cache'
  if (Test-Path $sceneCache) {
    Write-Host "Cleaning thumbnail scene cache..."
    Remove-Item $sceneCache -Recurse -Force
  }
}

Write-Host 'www build complete.'
