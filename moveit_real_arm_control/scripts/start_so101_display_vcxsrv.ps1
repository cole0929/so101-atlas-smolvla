$ErrorActionPreference = 'Stop'

# Start VcXsrv + SO-ARM101 display.launch.py (RViz) while WSLg is broken.
# Use this instead of "ros2 launch so101_description display.launch.py" in a
# WSLg terminal until a Windows reboot restores the WSLg GPU-PV path.

$vcxSrv = 'F:\VcXsrv\vcxsrv.exe'
$authWindows = 'F:\robot_arm_atlas\.x11\vcxsrv.Xauthority'
$prepareScript = '/mnt/f/robot_arm_atlas/ros2/prepare_vcxsrv_xauth.sh'
$displayScript = '/mnt/f/robot_arm_atlas/ros2/run_so101_display_vcxsrv.sh'

if (-not (Test-Path -LiteralPath $vcxSrv)) {
    throw "VcXsrv not found: $vcxSrv"
}

# Stop only this workspace's dedicated display :1 instance.
Get-CimInstance Win32_Process -Filter "Name='vcxsrv.exe'" |
    Where-Object { $_.ExecutablePath -eq $vcxSrv -and $_.CommandLine -match '(^|\s):1(\s|$)' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

$windowsHost = (& wsl.exe -d Ubuntu-22.04 -- bash $prepareScript | Where-Object { $_ -match '^\d+\.\d+\.\d+\.\d+$' } | Select-Object -Last 1)
if (-not $windowsHost) {
    throw 'Could not determine the Windows host address from WSL.'
}

$vcxArgs = @(
    ':1',
    '-multiwindow',
    '-clipboard',
    '-wgl',
    '-listen', 'tcp',
    '-auth', $authWindows,
    '-silent-dup-error'
)
Start-Process -FilePath $vcxSrv -ArgumentList $vcxArgs -WindowStyle Hidden

$ready = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    Start-Sleep -Milliseconds 200
    if (Get-NetTCPConnection -LocalPort 6001 -State Listen -ErrorAction SilentlyContinue) {
        $ready = $true
        break
    }
}
if (-not $ready) {
    throw 'VcXsrv did not start listening on display :1 (TCP 6001).'
}

Write-Host "VcXsrv is ready on $windowsHost`:1; starting SO-ARM101 display.launch.py (RViz)." -ForegroundColor Green
Write-Host 'Press Ctrl+C in this terminal to stop RViz.' -ForegroundColor Yellow
& wsl.exe -d Ubuntu-22.04 -- bash $displayScript
