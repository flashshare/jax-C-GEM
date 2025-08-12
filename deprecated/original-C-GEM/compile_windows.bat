@echo off
REM Windows-compatible compile script for C-GEM
REM This script only compiles, does not run the simulation

echo 🔧 Compiling Original C-GEM (Windows Compatible)...
echo ================================================

REM Try different GCC paths in order of preference
set GCC_FOUND=0

REM Check for Code::Blocks MinGW GCC
if exist "C:\Program Files\CodeBlocks\MinGW\bin\gcc.exe" (
    set "GCC_PATH=C:\Program Files\CodeBlocks\MinGW\bin\gcc.exe"
    set GCC_FOUND=1
    goto :compile
)

REM Check for standalone MinGW
if exist "C:\MinGW\bin\gcc.exe" (
    set "GCC_PATH=C:\MinGW\bin\gcc.exe"
    set GCC_FOUND=1
    goto :compile
)

REM Check for MSYS2 MinGW64
if exist "C:\msys64\mingw64\bin\gcc.exe" (
    set "GCC_PATH=C:\msys64\mingw64\bin\gcc.exe"
    set GCC_FOUND=1
    goto :compile
)

REM Try system PATH gcc
where gcc >nul 2>&1
if %errorlevel% equ 0 (
    set "GCC_PATH=gcc"
    set GCC_FOUND=1
    goto :compile
)

REM No GCC found
echo ❌ ERROR: GCC compiler not found!
echo.
echo Please install one of:
echo - Code::Blocks with MinGW
echo - Standalone MinGW 
echo - MSYS2
echo - Or add GCC to your PATH
exit /b 1

:compile
echo 🏗️ Using GCC: %GCC_PATH%

REM Create output directories
if not exist "OUT" mkdir OUT
if not exist "OUT\Hydrodynamics" mkdir OUT\Hydrodynamics
if not exist "OUT\Flux" mkdir OUT\Flux
if not exist "OUT\Reaction" mkdir OUT\Reaction

REM Compile with Windows-specific flags
echo 🏗️ Compiling C source files...
"%GCC_PATH%" -D_WIN32 -o c-gem.exe *.c -lm -O2

if %errorlevel% neq 0 (
    echo ❌ Compilation failed!
    exit /b 1
)

echo ✅ Compilation successful!
echo 📂 Executable created: c-gem.exe
echo 🎯 Ready for benchmark execution
