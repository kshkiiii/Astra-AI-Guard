@echo off
call "%~dp0START.bat" --build %*
exit /b %errorlevel%
