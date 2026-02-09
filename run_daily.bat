@echo off
setlocal

pushd "%~dp0"

set "PS_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
set "DEFAULT_ARGS=-Experiment mi_experimento -Top 0 -OpenNotional 250 -HoldingsLogMode latest -Fill next_open -SlippageBps 10 -Fee 0"

if "%~1"=="" (
  echo Ejecutando con parametros por defecto...
  "%PS_EXE%" -NoProfile -ExecutionPolicy Bypass -File ".\run_daily.ps1" %DEFAULT_ARGS%
) else (
  echo Ejecutando con parametros personalizados...
  "%PS_EXE%" -NoProfile -ExecutionPolicy Bypass -File ".\run_daily.ps1" %*
)

if errorlevel 1 goto :error
goto :ok

:error
echo.
echo El proceso termino con error (code=%errorlevel%).
echo Tip: si ya corriste hoy y quieres repetir, agrega -ForcePaperTrade.
pause
popd
exit /b %errorlevel%

:ok
popd
exit /b 0
