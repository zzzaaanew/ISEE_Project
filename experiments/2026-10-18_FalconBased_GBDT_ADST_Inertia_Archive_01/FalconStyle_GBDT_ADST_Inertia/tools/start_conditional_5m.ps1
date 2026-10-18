$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskWorkspace = Split-Path -Parent $taskRoot
$taskPython = Join-Path $taskWorkspace 'Falcon_AcmeTrace_ST_Top100/.venv/Scripts/python.exe'
$taskWatcher = Join-Path $PSScriptRoot 'conditional_5m.py'
$taskTarget = Join-Path $taskRoot 'runs/conditional_5m_01'
$taskSource = Join-Path $taskRoot 'runs/full_01'
if (Test-Path -LiteralPath (Join-Path $taskTarget 'watcher_status.json')) {
    throw 'An existing conditional watcher has already claimed this output. Inspect its status first.'
}
New-Item -ItemType Directory -Path $taskTarget -Force | Out-Null
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$taskArguments = @('-u', ('"' + $taskWatcher + '"'), '--source', ('"' + $taskSource + '"'),
    '--target', ('"' + $taskTarget + '"'), '--wait-supervisor-pid', '22224',
    '--max-wait-seconds', '43200', '--max-run-seconds', '43200')
$taskProcess = Start-Process -FilePath $taskPython -ArgumentList $taskArguments -WorkingDirectory $taskWorkspace `
    -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskTarget 'watcher_stdout.log') `
    -RedirectStandardError (Join-Path $taskTarget 'watcher_stderr.log') -PassThru
[pscustomobject]@{launcher_pid=$taskProcess.Id; output=$taskTarget; source=$taskSource} | ConvertTo-Json
