@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto existing
where py >nul 2>nul
if not errorlevel 1 goto launcher
where python >nul 2>nul
if not errorlevel 1 goto python
echo Install Python 3.12 x64 from https://www.python.org/downloads/windows/
pause
exit /b 1
:existing
".venv\Scripts\python.exe" -X utf8 tools\launch.py %*
goto done
:launcher
py -3.12 -X utf8 tools\launch.py %*
goto done
:python
python -X utf8 tools\launch.py %*
:done
if errorlevel 1 goto failed
exit /b 0
:failed
echo Startup failed. See the error above.
pause
exit /b 1
