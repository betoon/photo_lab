$ErrorActionPreference = "Stop"
$packRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$toolsRoot = Join-Path $packRoot "tools"
$modelsRoot = Join-Path $packRoot "models\ddcolor_paper_tiny"
$archive = Join-Path $env:TEMP "photolab-ddcolor-source.zip"
$staging = Join-Path $env:TEMP "photolab-ddcolor-staging"
$sourceDestination = Join-Path $toolsRoot "ddcolor"
$sourceUrl = "https://github.com/piddnad/DDColor/archive/refs/heads/master.zip"
$weightsUrl = "https://huggingface.co/piddnad/ddcolor_paper_tiny/resolve/main/pytorch_model.bin"
$configUrl = "https://huggingface.co/piddnad/ddcolor_paper_tiny/resolve/main/config.json"
$expectedSha256 = "8a1277bc90a1bfbb6d2d83933a9a6bc821931879ca93e26e4fcec12165d41fce"

Write-Host "DDColor local provider installer"
Write-Host "Source and checkpoint license: Apache-2.0"
Write-Host "Downloading official DDColor source..."
Invoke-WebRequest -Uri $sourceUrl -OutFile $archive
if (Test-Path -LiteralPath $staging) { Remove-Item -LiteralPath $staging -Recurse -Force }
New-Item -ItemType Directory -Path $staging | Out-Null
Expand-Archive -LiteralPath $archive -DestinationPath $staging -Force
if (Test-Path -LiteralPath $sourceDestination) { Remove-Item -LiteralPath $sourceDestination -Recurse -Force }
$source = Get-ChildItem -LiteralPath $staging -Directory | Select-Object -First 1
if (-not $source) { throw "The official archive did not contain DDColor source." }
Move-Item -LiteralPath $source.FullName -Destination $sourceDestination

New-Item -ItemType Directory -Force -Path $modelsRoot | Out-Null
$weights = Join-Path $modelsRoot "pytorch_model.bin"
Write-Host "Downloading the official DDColor paper-tiny checkpoint (about 220 MB)..."
Invoke-WebRequest -Uri $weightsUrl -OutFile $weights
Invoke-WebRequest -Uri $configUrl -OutFile (Join-Path $modelsRoot "config.json")
$actualSha256 = (Get-FileHash -LiteralPath $weights -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualSha256 -ne $expectedSha256) {
    Remove-Item -LiteralPath $weights -Force
    throw "DDColor checkpoint verification failed. Expected $expectedSha256, received $actualSha256."
}

if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force }
if (Test-Path -LiteralPath $staging) { Remove-Item -LiteralPath $staging -Recurse -Force }
Write-Host "DDColor installation complete and SHA-256 verified."
& python (Join-Path $packRoot "diagnose_model_pack.py")
