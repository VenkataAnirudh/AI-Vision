@echo off
setlocal EnableExtensions EnableDelayedExpansion
title VisionAI Intelligence Engine v3.0
color 0B

set "DEBUG_LOG=debug.txt"
call :log "============================================================"
call :log "VisionAI launcher started. cwd=%CD%"

echo ============================================================
echo       VisionAI Intelligence Engine v3.0
echo       Production-Grade Surveillance Analytics
echo ============================================================
echo.

set "PYTHON_CMD="
if exist "venv1\Scripts\python.exe" set "PYTHON_CMD=%CD%\venv1\Scripts\python.exe"
if not defined PYTHON_CMD if exist "venv\Scripts\python.exe" set "PYTHON_CMD=%CD%\venv\Scripts\python.exe"
if not defined PYTHON_CMD if exist ".venv\Scripts\python.exe" set "PYTHON_CMD=%CD%\.venv\Scripts\python.exe"
if not defined PYTHON_CMD set "PYTHON_CMD=python"

call :log "Selected Python command: %PYTHON_CMD%"
call :log "CMD start: ""%PYTHON_CMD%"" --version"
"%PYTHON_CMD%" --version >> "%DEBUG_LOG%" 2>&1
if errorlevel 1 (
    call :log "Python version check failed. The selected Python command is not usable."
    echo [ERROR] Python is not usable. Check debug.txt for details.
    pause
    exit /b 1
)

echo [0/4] Clearing ports 8000 and 8501...
call :log "Clearing ports 8000 and 8501"

for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8000 " ^| findstr "LISTENING"') do (
    if not "%%a"=="0" (
        echo    Killing PID %%a on port 8000...
        call :log "CMD start: taskkill /F /PID %%a"
        taskkill /F /PID %%a >> "%DEBUG_LOG%" 2>&1
    )
)

for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8501 " ^| findstr "LISTENING"') do (
    if not "%%a"=="0" (
        echo    Killing PID %%a on port 8501...
        call :log "CMD start: taskkill /F /PID %%a"
        taskkill /F /PID %%a >> "%DEBUG_LOG%" 2>&1
    )
)

timeout /t 1 /nobreak > nul
echo    Ports cleared.
call :log "Ports cleared"
echo.

echo [1/4] Setting up directory structure...
call :log "Ensuring output and upload directories exist"
if not exist "output" mkdir output
if not exist "output\reports" mkdir output\reports
if not exist "output\annotated" mkdir output\annotated
if not exist "output\events" mkdir output\events
if not exist "uploads" mkdir uploads

echo [2/4] Starting FastAPI backend (port 8000)...
call :log "CMD start: ""%PYTHON_CMD%"" api/server.py > output\server.log 2>&1"
start /B "" "%PYTHON_CMD%" api/server.py > output\server.log 2>&1

echo [3/4] Waiting for backend to become ready...
call :log "Waiting for backend health endpoint"
set RETRIES=0
set MAX_RETRIES=30

:health_loop
if %RETRIES% GEQ %MAX_RETRIES% (
    echo.
    echo [ERROR] Backend did not start within 60 seconds.
    call :log "Backend did not start within 60 seconds"
    echo.
    echo ---- Last 20 lines of server.log ----
    powershell -NoProfile -Command "Get-Content output\server.log -Tail 20" 2>nul
    echo ---- End of server.log ----
    call :log "Appending server.log tail after backend startup failure"
    powershell -NoProfile -Command "Add-Content -Path '%DEBUG_LOG%' -Value ('[{0}] [run.bat] ---- Last 80 lines of output/server.log ----' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')); if (Test-Path 'output/server.log') { Get-Content 'output/server.log' -Tail 80 | ForEach-Object { Add-Content -Path '%DEBUG_LOG%' -Value ('[{0}] [server.log] {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $_) } }" 2>nul
    echo.
    echo Common fixes:
    echo   "%PYTHON_CMD%" -m pip install -r requirements.txt
    echo   "%PYTHON_CMD%" -m pip install supervision
    echo.
    pause
    exit /b 1
)

timeout /t 2 /nobreak > nul
set /a RETRIES+=1

"%PYTHON_CMD%" -c "import requests; r=requests.get('http://localhost:8000/health', timeout=2); exit(0 if r.status_code==200 else 1)" >> "%DEBUG_LOG%" 2>&1
if %errorlevel% neq 0 (
    echo    Attempt %RETRIES%/%MAX_RETRIES% - waiting...
    call :log "Health check attempt %RETRIES%/%MAX_RETRIES% failed"
    goto health_loop
)

echo    Backend is READY!
call :log "Backend health check passed"
echo.

echo [4/4] Launching Streamlit Dashboard...
echo.
echo ============================================================
echo    Dashboard: http://localhost:8501
echo    API Docs:  http://localhost:8000/docs
echo    Health:    http://localhost:8000/health
echo    Debug:     debug.txt
echo ============================================================
echo.

call :log "Opening dashboard URL in default browser"

call :log "CMD start: ""%PYTHON_CMD%"" -m streamlit run ui/dashboard.py"
powershell -NoProfile -ExecutionPolicy Bypass -Command "& $env:PYTHON_CMD -m streamlit run 'ui/dashboard.py' 2>&1 | ForEach-Object { $line = $_.ToString(); Add-Content -Path '%DEBUG_LOG%' -Value ('[{0}] [streamlit] {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $line); Write-Host $line }; exit $LASTEXITCODE"
set "EXIT_CODE=%ERRORLEVEL%"
call :log "Streamlit exited with code %EXIT_CODE%"
exit /b %EXIT_CODE%

:log
>> "%DEBUG_LOG%" echo [%DATE% %TIME%] [run.bat] %~1
exit /b 0
