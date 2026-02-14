param(
    [string]$Experiment = "mi_experimento",
    [string]$PythonPath = ".\.venv311\Scripts\python.exe",
    [string]$HoldingsPath = ".\config\holdings.txt",
    [string]$RunTag = "",
    [int]$Top = 0,
    [int]$HoldingsLookbackYears = 5,
    [int]$BacktestDays = 252,
    [double]$InitialCapital = 100.0,
    [switch]$PlotHoldings,
    [double]$OpenNotional = 250.0,
    [ValidateSet("latest", "date", "all")][string]$HoldingsLogMode = "latest",
    [string]$HoldingsLogDate = "",
    [ValidateSet("next_open", "next_close")][string]$Fill = "next_open",
    [double]$SlippageBps = 10.0,
    [double]$Fee = 0.0,
    [switch]$ForcePaperTrade
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-AsOfString {
    param([datetime]$NowDate)
    $d = $NowDate.Date
    if ($d.DayOfWeek -eq [System.DayOfWeek]::Saturday) {
        $d = $d.AddDays(-1)
    } elseif ($d.DayOfWeek -eq [System.DayOfWeek]::Sunday) {
        $d = $d.AddDays(-2)
    }
    return $d.ToString("yyyy-MM-dd")
}

if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "No existe Python en: $PythonPath"
}

$dataDir = Join-Path -Path "Experiments" -ChildPath (Join-Path -Path $Experiment -ChildPath "csv_files")
$dailyCsv = Join-Path -Path $dataDir -ChildPath "Daily Updates.csv"
$tradeCsv = Join-Path -Path $dataDir -ChildPath "Trade Log.csv"

if (-not (Test-Path -LiteralPath $dailyCsv) -or -not (Test-Path -LiteralPath $tradeCsv)) {
    throw "No existen CSVs del experimento en $dataDir. Inicializa con: .\init_experiment.py --name $Experiment --starting-equity 10000"
}

$asof = Get-AsOfString -NowDate (Get-Date)
$runDir = Join-Path -Path "runs" -ChildPath $asof
$runTagTrim = $RunTag.Trim()
if (-not [string]::IsNullOrWhiteSpace($runTagTrim)) {
    $runDir = Join-Path -Path $runDir -ChildPath $runTagTrim
}
$holdingsTradeLogPath = Join-Path -Path $runDir -ChildPath "holdings_trade_log.csv"

Write-Host "AsOf: $asof"
Write-Host "Experimento: $Experiment"
Write-Host "Data dir: $dataDir"
if (-not [string]::IsNullOrWhiteSpace($runTagTrim)) {
    Write-Host "RunTag: $runTagTrim"
}

$recommendArgs = @(
    ".\recommend.py",
    "--non-interactive",
    "--asof", $asof,
    "--top", "$Top",
    "--holdings", $HoldingsPath,
    "--holdings-lookback-years", "$HoldingsLookbackYears",
    "--backtest-days", "$BacktestDays",
    "--initial-capital", "$InitialCapital"
)

if (-not [string]::IsNullOrWhiteSpace($runTagTrim)) {
    $recommendArgs += @("--run-tag", $runTagTrim)
}

if ($PlotHoldings.IsPresent) {
    $recommendArgs += "--plot-holdings"
}

Write-Host "`n[1/2] Ejecutando recommend.py ..."
& $PythonPath @recommendArgs
if ($LASTEXITCODE -ne 0) {
    throw "recommend.py fallo con exit code $LASTEXITCODE"
}

if (-not (Test-Path -LiteralPath $holdingsTradeLogPath)) {
    throw "No se genero holdings_trade_log.csv en: $holdingsTradeLogPath"
}

$paperArgs = @(
    ".\paper_trade.py",
    "--data-dir", $dataDir,
    "--holdings-trade-log", $holdingsTradeLogPath,
    "--holdings-log-mode", $HoldingsLogMode,
    "--open-notional", "$OpenNotional",
    "--fill", $Fill,
    "--slippage-bps", "$SlippageBps",
    "--fee", "$Fee"
)

if ($HoldingsLogMode -eq "date" -and -not [string]::IsNullOrWhiteSpace($HoldingsLogDate)) {
    $paperArgs += @("--holdings-log-date", $HoldingsLogDate)
}

if ($ForcePaperTrade.IsPresent) {
    $paperArgs += "--force"
}

Write-Host "`n[2/2] Ejecutando paper_trade.py ..."
& $PythonPath @paperArgs
if ($LASTEXITCODE -ne 0) {
    throw "paper_trade.py fallo con exit code $LASTEXITCODE"
}

Write-Host "`nListo."
Write-Host "- Holdings trade log: $holdingsTradeLogPath"
Write-Host "- Daily Updates: $dailyCsv"
Write-Host "- Trade Log: $tradeCsv"
if ($Top -gt 0) {
    Write-Host "- Nota: Top > 0 solo genera ranking informativo en modo no interactivo."
}
