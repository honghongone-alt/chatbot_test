@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 goto folder_error

rem Reuse the local Streamlit server if it is already running.
powershell.exe -NoProfile -NonInteractive -WindowStyle Hidden -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8501/_stcore/health' -TimeoutSec 2; if ($r.StatusCode -eq 200 -and $r.Content.Trim() -eq 'ok') { exit 0 }; exit 1 } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 (
    start "" "http://127.0.0.1:8501"
    exit /b 0
)

where uv >nul 2>&1
if errorlevel 1 (
    echo ERROR: uv was not found. Install uv and try again.
    pause
    exit /b 1
)

if not exist "app.py" (
    echo ERROR: app.py was not found in the project folder.
    pause
    exit /b 1
)

if not exist ".env" (
    echo ERROR: Create .env and set OPENAI_API_KEY before running.
    pause
    exit /b 1
)

set "PYTHONUTF8=1"
echo Starting the chatbot. Keep this window open while using it.
echo Press Ctrl+C to stop the server.
rem --no-editable avoids Python 3.11 encoding errors in Korean folder paths.
uv run --locked --no-editable streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless false --browser.gatherUsageStats false
if errorlevel 1 (
    echo.
    echo ERROR: The chatbot could not start. Check the message above.
    pause
    exit /b 1
)
exit /b 0

:folder_error
echo ERROR: Could not open the project folder.
pause
exit /b 1
