@echo off
cd /d "%~dp0"
set "PY_EXE=.venv\Scripts\python.exe"
if not exist "%PY_EXE%" set "PY_EXE=.venv311\Scripts\python.exe"

if not exist "%PY_EXE%" (
	echo Python sanal ortami bulunamadi: .venv veya .venv311
	pause
	exit /b 1
)

"%PY_EXE%" "app.py"
if errorlevel 1 (
	echo.
	echo Uygulama hata ile kapandi. Yukaridaki hatayi kontrol et.
	pause
)
