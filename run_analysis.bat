@echo off
REM ============================================================================
REM THz-ISL MIMO ISAC Analysis Framework - Windows Automation Script
REM DR-08 Protocol Compliant Execution Sequence
REM ============================================================================

echo ============================================================================
echo THz-ISL MIMO ISAC PERFORMANCE ANALYSIS FRAMEWORK
echo ============================================================================
echo.

REM 设置Python解释器(根据你的环境修改)
set PYTHON=python

REM 检查Python是否可用
%PYTHON% --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.8+ and add to PATH
    pause
    exit /b 1
)

echo [INFO] Python found:
%PYTHON% --version
echo.

REM ============================================================================
REM STEP 0: Integration Test (Optional but Recommended)
REM ============================================================================
echo ============================================================================
echo STEP 0: Running Integration Test (验证系统完整性)
echo ============================================================================
echo.

%PYTHON% test_integration.py
if errorlevel 1 (
    echo.
    echo [ERROR] Integration test failed! Please fix errors before continuing.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Integration test passed!
echo.
pause

REM ============================================================================
REM STEP 1: Main Pareto Analysis
REM ============================================================================
echo ============================================================================
echo STEP 1: Running Main Pareto Analysis (主分析:生成Pareto前沿)
echo ============================================================================
echo.

%PYTHON% main.py config.yaml
if errorlevel 1 (
    echo.
    echo [ERROR] Main analysis failed!
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Main analysis completed!
echo.
pause

REM ============================================================================
REM STEP 2: SNR Sweep Analysis
REM ============================================================================
echo ============================================================================
echo STEP 2: Running SNR Sweep Analysis (SNR扫描分析)
echo ============================================================================
echo.

%PYTHON% scan_snr_sweep.py config.yaml
if errorlevel 1 (
    echo.
    echo [ERROR] SNR sweep failed!
    pause
    exit /b 1
)

echo.
echo [SUCCESS] SNR sweep completed!
echo.
pause

REM ============================================================================
REM STEP 3: Threshold Validation (Optional)
REM ============================================================================
echo ============================================================================
echo STEP 3: Running Threshold Validation (阈值验证,可选)
echo ============================================================================
echo.

set /p RUN_THRESHOLD="Run threshold validation? (y/n, 耗时较长): "
if /i "%RUN_THRESHOLD%"=="y" (
    %PYTHON% threshold_sweep.py config.yaml
    if errorlevel 1 (
        echo.
        echo [WARNING] Threshold validation failed, but continuing...
    ) else (
        echo.
        echo [SUCCESS] Threshold validation completed!
    )
    echo.
    pause
)

REM ============================================================================
REM STEP 4: Visualization
REM ============================================================================
echo ============================================================================
echo STEP 4: Generating Visualizations (生成图表)
echo ============================================================================
echo.

%PYTHON% visualize_results.py
if errorlevel 1 (
    echo.
    echo [ERROR] Visualization failed!
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Visualizations generated!
echo.
pause

REM ============================================================================
REM STEP 5: Generate Paper Tables
REM ============================================================================
echo ============================================================================
echo STEP 5: Generating Paper Tables (生成论文表格)
echo ============================================================================
echo.

%PYTHON% make_paper_tables.py
if errorlevel 1 (
    echo.
    echo [ERROR] Table generation failed!
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Paper tables generated!
echo.

REM ============================================================================
REM COMPLETION SUMMARY
REM ============================================================================
echo ============================================================================
echo ANALYSIS COMPLETE - ALL STEPS FINISHED SUCCESSFULLY
echo ============================================================================
echo.
echo Generated outputs:
echo   - Pareto results:     results\DR08_improved_pareto_results.csv
echo   - SNR sweep:          results\DR08_improved_snr_sweep.csv
echo   - Figures:            figures\
echo   - Paper tables:       (printed to console)
echo.
echo Next steps:
echo   1. Check figures\ directory for plots
echo   2. Review LaTeX tables in console output
echo   3. Check results\ directory for CSV data
echo.
echo ============================================================================
pause