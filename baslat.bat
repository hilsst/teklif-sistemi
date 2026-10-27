@echo off
REM Teklif sistemi: Windows baslatici. Cift tiklayin.
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
  echo uv kurulu degil. PowerShell ile kurun:
  echo   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  pause
  exit /b 1
)
uv sync --extra app
if errorlevel 1 (
  pause
  exit /b 1
)
uv run teklif app
pause
