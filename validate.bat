@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
where uv >nul 2>&1
if errorlevel 1 (
    echo ERROR: uv was not found.
    pause
    exit /b 1
)
if /i "%~1"=="offline" goto offline
echo Validating extraction, retrieval and 14 answers using OpenAI API.
echo API usage charges apply. Chat history will be preserved.
uv run --locked --no-editable python -X utf8 -W error -u scripts\validate_rag.py --live
set "validation_exit=%errorlevel%"
if exist ".chatbot\validation_latest.html" start "" ".chatbot\validation_latest.html"
if not "%validation_exit%"=="0" echo Some checks need review. See the report.
pause
exit /b %validation_exit%

:offline
echo Validating extraction without API calls.
uv run --locked --no-editable python -X utf8 -W error -u scripts\validate_rag.py --offline
set "validation_exit=%errorlevel%"
if exist ".chatbot\extraction_latest.html" start "" ".chatbot\extraction_latest.html"
pause
exit /b %validation_exit%
