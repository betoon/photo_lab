param([string]$SdkRoot)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
if (-not $SdkRoot) { $SdkRoot = Join-Path (Split-Path $repoRoot -Parent) 'nikon_sdk\Image SDK' }
$headerDir = Join-Path $SdkRoot 'Library\win\Include'
if (-not (Test-Path (Join-Path $headerDir 'Nkfl_Interface.h'))) { throw 'Pass -SdkRoot pointing to the Nikon Image SDK folder.' }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installation) { throw 'Install Visual Studio C++ build tools to build the Nikon helper.' }
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
$buildDir = Join-Path $repoRoot 'build\nikon_decoder'
New-Item -ItemType Directory -Force $buildDir | Out-Null
$source = Join-Path $PSScriptRoot 'nikon_decoder.cpp'
$exe = Join-Path $PSScriptRoot 'nikon_decoder.exe'
$command = Join-Path $buildDir 'compile.cmd'
@"
@echo off
call "$vcvars" >nul
if errorlevel 1 exit /b 1
cl /nologo /std:c++17 /EHsc /O2 /MT /I "$headerDir" "$source" /Fe:"$exe" /Fo:"$buildDir\nikon_decoder.obj"
"@ | Set-Content $command -Encoding ascii
& $env:ComSpec /c $command
if ($LASTEXITCODE -ne 0) { throw 'Nikon helper build failed.' }
Write-Output "Built $exe"
