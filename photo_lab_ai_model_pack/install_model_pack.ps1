$ErrorActionPreference = "Stop"
$packRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$toolsRoot = Join-Path $packRoot "tools"
$archive = Join-Path $env:TEMP "photolab-realesrgan-windows.zip"
$destination = Join-Path $toolsRoot "realesrgan-ncnn-vulkan"
$download = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip"

Write-Host "PhotoLab Local AI Model Pack installer"
Write-Host "Downloading the official Real-ESRGAN v0.2.5.0 Windows package..."
Invoke-WebRequest -Uri $download -OutFile $archive

$staging = Join-Path $env:TEMP "photolab-realesrgan-staging"
if (Test-Path -LiteralPath $staging) { Remove-Item -LiteralPath $staging -Recurse -Force }
New-Item -ItemType Directory -Path $staging | Out-Null
Expand-Archive -LiteralPath $archive -DestinationPath $staging -Force

if (Test-Path -LiteralPath $destination) { Remove-Item -LiteralPath $destination -Recurse -Force }
$engine = Get-ChildItem -LiteralPath $staging -Recurse -Filter "realesrgan-ncnn-vulkan.exe" | Select-Object -First 1
if (-not $engine) { throw "The official archive did not contain realesrgan-ncnn-vulkan.exe" }
Move-Item -LiteralPath $engine.Directory.FullName -Destination $destination
if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force }
if (Test-Path -LiteralPath $staging) { Remove-Item -LiteralPath $staging -Recurse -Force }

Write-Host "Installation complete. Running diagnostics..."
& python (Join-Path $packRoot "diagnose_model_pack.py")
Write-Host "In PhotoLab, set AI Restoration Model Pack to:"
Write-Host $packRoot
