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

Write-Host 'Building wasm (web feature set)...'
Invoke-Step cargo build --target wasm32-unknown-unknown --release --no-default-features --features web

$wasmPath = Join-Path 'target/wasm32-unknown-unknown/release' 'bevy_gaussian_splatting.wasm'
if (-not (Test-Path $wasmPath)) {
  throw "wasm output not found at $wasmPath"
}

if (-not (Get-Command wasm-bindgen -ErrorAction SilentlyContinue)) {
  throw 'wasm-bindgen not found on PATH'
}

Write-Host 'Generating wasm bindings...'
Invoke-Step wasm-bindgen --out-dir ./www/out --target web $wasmPath

Write-Host 'Rendering example thumbnails from manifest...'
$env:RENDER_EXAMPLE_THUMBNAILS = '1'
try {
  Invoke-Step cargo test --test headless_examples render_example_thumbnails -- --nocapture
} finally {
  Remove-Item Env:RENDER_EXAMPLE_THUMBNAILS -ErrorAction SilentlyContinue
}

Write-Host 'www build complete.'
