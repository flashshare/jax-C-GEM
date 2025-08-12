@echo off
REM Compilation and execution script for original C-GEM
REM This script compiles the C source files and runs the simulation

echo 🔧 Compiling Original C-GEM...
echo ==========================================

REM Use Code::Blocks MinGW GCC compiler
set GCC_PATH="C:/Program Files/CodeBlocks/MinGW/bin/gcc.exe"

REM Check if Code::Blocks GCC is available
%GCC_PATH% --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ERROR: Code::Blocks GCC compiler not found!
    echo Please check if Code::Blocks is installed with MinGW at:
    echo "C:/Program Files/CodeBlocks/MinGW/bin/gcc.exe"
    pause
    exit /b 1
)

REM Create OUT directories if they don't exist
if not exist "OUT" mkdir OUT
if not exist "OUT\Hydrodynamics" mkdir OUT\Hydrodynamics
if not exist "OUT\Flux" mkdir OUT\Flux
if not exist "OUT\Reaction" mkdir OUT\Reaction

REM Compile the C-GEM model
echo 🏗️ Compiling C source files...
%GCC_PATH% -o c-gem.exe *.c -lm -O2

if %errorlevel% neq 0 (
    echo ❌ Compilation failed!
    pause
    exit /b 1
)

echo ✅ Compilation successful!


echo ✅ Compilation completed!
