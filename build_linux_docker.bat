@echo off
setlocal
cd /d "%~dp0"

echo === PhotoLab Linux x86_64 build ===
docker info >nul 2>&1
if errorlevel 1 (
  echo ERROR: Docker Desktop is not running.
  exit /b 1
)

docker build --pull -f Dockerfile.linux-build -t photolab-linux-builder .
if errorlevel 1 exit /b 1
for /f %%i in ('docker create photolab-linux-builder') do set "PHOTOLAB_CONTAINER=%%i"
if not defined PHOTOLAB_CONTAINER exit /b 1
docker cp "%PHOTOLAB_CONTAINER%:/src/PhotoLab-linux-x86_64.tar.gz" .
set "PHOTOLAB_COPY_RESULT=%ERRORLEVEL%"
docker rm "%PHOTOLAB_CONTAINER%" >nul
if not "%PHOTOLAB_COPY_RESULT%"=="0" exit /b %PHOTOLAB_COPY_RESULT%

echo Output: PhotoLab-linux-x86_64.tar.gz
endlocal
