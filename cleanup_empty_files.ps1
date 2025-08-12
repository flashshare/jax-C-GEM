#!/usr/bin/env powershell
# Cleanup Empty Files Script for C-GEM Project
# This script removes all empty files (0 KB) from the workspace
# Run with: powershell -ExecutionPolicy Bypass .\cleanup_empty_files.ps1

$workspacePath = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "🧹 Cleaning up empty files in: $workspacePath" -ForegroundColor Cyan

# Find all empty files
$emptyFiles = Get-ChildItem -Path $workspacePath -Recurse -File | Where-Object { $_.Length -eq 0 }

if ($emptyFiles.Count -eq 0) {
    Write-Host "✅ No empty files found! Workspace is clean." -ForegroundColor Green
} else {
    Write-Host "🗑️  Found $($emptyFiles.Count) empty files:" -ForegroundColor Yellow
    $emptyFiles | ForEach-Object { Write-Host "   - $($_.FullName)" -ForegroundColor Gray }
    
    # Remove empty files
    $emptyFiles | Remove-Item -Force -Verbose
    
    Write-Host "✅ Successfully removed $($emptyFiles.Count) empty files!" -ForegroundColor Green
}

# Verify cleanup
$remainingEmpty = Get-ChildItem -Path $workspacePath -Recurse -File | Where-Object { $_.Length -eq 0 }
if ($remainingEmpty.Count -eq 0) {
    Write-Host "🎉 Workspace cleanup complete!" -ForegroundColor Green
} else {
    Write-Host "⚠️  Warning: $($remainingEmpty.Count) empty files still remain" -ForegroundColor Yellow
}
