@echo off
REM Setup script for original C-GEM
REM This script copies the INPUT directory and sets up the environment

echo 🔧 Setting up Original C-GEM Environment...
echo ================================================

REM Check if we're in the right directory
if not exist "main.c" (
    echo ❌ ERROR: Please run this script from the original-C-GEM directory
    echo Current directory should contain main.c, define.h, etc.
    pause
    exit /b 1
)

REM Copy INPUT directory from parent JAX C-GEM directory
echo 📁 Setting up INPUT directory...
if exist "..\..\INPUT" (
    echo    Copying INPUT directory from JAX C-GEM...
    xcopy "..\..\INPUT" "INPUT\" /E /I /Y
    echo    ✅ INPUT directory copied successfully
) else (
    echo    ❌ ERROR: INPUT directory not found in JAX C-GEM root
    echo    Please ensure the JAX C-GEM INPUT directory exists
    pause
    exit /b 1
)

REM Create OUTPUT directories
echo 📁 Creating OUTPUT directories...
if not exist "OUT" mkdir OUT
if not exist "OUT\Hydrodynamics" mkdir OUT\Hydrodynamics
if not exist "OUT\Flux" mkdir OUT\Flux
if not exist "OUT\Reaction" mkdir OUT\Reaction
echo    ✅ Output directories created

REM Check for required input files
echo 🔍 Checking required input files...
set /a missing_files=0

REM Check essential files
if not exist "INPUT\Boundary\wind.csv" (
    echo    ❌ Missing: INPUT\Boundary\wind.csv
    set /a missing_files+=1
)
if not exist "INPUT\Boundary\UB\discharge_ub.csv" (
    echo    ❌ Missing: INPUT\Boundary\UB\discharge_ub.csv
    set /a missing_files+=1
)
if not exist "INPUT\Boundary\LB\elevation.csv" (
    echo    ❌ Missing: INPUT\Boundary\LB\elevation.csv
    set /a missing_files+=1
)

if %missing_files% gtr 0 (
    echo    ⚠️  Warning: %missing_files% required files are missing
    echo    The simulation may fail without these files
) else (
    echo    ✅ All essential input files found
)

echo 
echo 🎯 Setup completed!
echo 
echo Next steps:
echo 1. Run: compile_and_run.bat
echo 2. Or manually: gcc -o c-gem *.c -lm -O2 && c-gem
echo 
echo Expected output: CSV files in OUT/ directory
pause
