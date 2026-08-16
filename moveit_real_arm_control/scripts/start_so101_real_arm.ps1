$ErrorActionPreference = 'Stop'

# ONE-CLICK PC launcher: VcXsrv + real_arm.launch.py (RViz + move_group).
# Run this AFTER the board one-click script (start_control_mode.sh).
#
#   powershell -ExecutionPolicy Bypass -File F:\robot_arm_atlas\ros2\start_so101_real_arm.ps1

$vcxSrv = 'F:\VcXsrv\vcxsrv.exe'
$authWindows = 'F:\robot_arm_atlas\.x11\vcxsrv.Xauthority'
$prepareScript = '/mnt/f/robot_arm_atlas/ros2/prepare_vcxsrv_xauth.sh'
$realArmScript = '/mnt/f/robot_arm_atlas/ros2/run_so101_real_arm_vcxsrv.sh'

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

Write-Host "VcXsrv ready on $windowsHost`:1; starting real-arm MoveIt (RViz + move_group)..." -ForegroundColor Green
Write-Host 'Press Ctrl+C in this terminal to stop.' -ForegroundColor Yellow
& wsl.exe -d Ubuntu-22.04 -- bash $realArmScript
