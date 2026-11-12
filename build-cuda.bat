@echo off
setlocal EnableExtensions DisableDelayedExpansion
if "%~1"=="--help" goto help
if "%~1"=="-h" goto help
if not "%~2"=="" goto bad_args
if not "%~1"=="" if not "%~1"=="--configure-only" goto bad_args
set "mode=%~1"

rem Use the same encoding for CMake and Ninja when they read MSVC header paths.
set "saved_cp="
for /f "tokens=2 delims=:" %%i in ('chcp') do set "saved_cp=%%i"
if defined saved_cp chcp 65001 >nul
call :build
set "build_status=%errorlevel%"
if defined saved_cp chcp %saved_cp% >nul
exit /b %build_status%

:build
if not defined JOBS set "JOBS=24"
if not defined CUDA_ARCHS set "CUDA_ARCHS=native"
powershell.exe -NoProfile -Command "if ($env:JOBS -cnotmatch '^[1-9][0-9]*$') { exit 2 }"
if errorlevel 1 (
    echo JOBS must be a positive integer. 1>&2
    exit /b 2
)

if not defined CUDAToolkit_ROOT set "CUDAToolkit_ROOT=%CUDA_PATH%"
if defined CUDAToolkit_ROOT set "PATH=%CUDAToolkit_ROOT%\bin;%PATH%"
where cl.exe >nul 2>nul
if not errorlevel 1 goto configure
set "vswhere=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%vswhere%" goto missing_msvc
set "vs_dir="
for /f "usebackq delims=" %%i in (`"%vswhere%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "vs_dir=%%i"
if not defined vs_dir goto missing_msvc
call "%vs_dir%\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 -no_logo
if errorlevel 1 exit /b 1

:configure
for %%i in (cmake.exe ninja.exe nvcc.exe) do (
    where %%i >nul 2>nul || (
        echo Missing tool: %%i 1>&2
        exit /b 1
    )
)
set "build_dir=%~dp0build-cuda"
echo Configuring CUDA Release build for architectures "%CUDA_ARCHS%"
cmake -S "%~dp0." -B "%build_dir%" -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_ARCHITECTURES=%CUDA_ARCHS%" ^
    -DGGML_CUDA=ON -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON ^
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF ^
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
if errorlevel 1 exit /b %errorlevel%
if "%mode%"=="--configure-only" exit /b 0

cmake --build "%build_dir%" --config Release --parallel "%JOBS%" --target llama-server llama-cli llama-bench llama-moe-autotune
if errorlevel 1 exit /b %errorlevel%
echo Build complete: "%build_dir%\bin"
exit /b 0

:missing_msvc
echo Install the Visual Studio x64 C++ build tools or run from an x64 Developer PowerShell. 1>&2
exit /b 1

:bad_args
echo Invalid arguments. Use --help. 1>&2
exit /b 2

:help
echo Usage: .\build-cuda.bat [--configure-only]
echo Builds llama-server, llama-cli, llama-bench and llama-moe-autotune in build-cuda\bin.
echo JOBS defaults to 24. CUDA_ARCHS defaults to native.
exit /b 0
