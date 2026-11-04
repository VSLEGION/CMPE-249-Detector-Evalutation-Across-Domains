param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$PipelineArgs = @("doctor")
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$searchDirectory = Get-Item -LiteralPath $repoRoot
$experimentPython = $null
while ($null -ne $searchDirectory) {
    $candidate = Join-Path $searchDirectory.FullName "dtect\Scripts\python.exe"
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        $experimentPython = $candidate
        break
    }
    $searchDirectory = $searchDirectory.Parent
}
if (-not $experimentPython) {
    throw "No ancestor dtect environment found. Activate a CUDA environment and run python run_pipeline.py."
}
& $experimentPython (Join-Path $repoRoot "run_pipeline.py") @PipelineArgs
exit $LASTEXITCODE
