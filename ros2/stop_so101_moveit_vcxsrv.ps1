$ErrorActionPreference = 'SilentlyContinue'

wsl.exe -d Ubuntu-22.04 -- bash /mnt/f/robot_arm_atlas/ros2/stop_so101_moveit_demo.sh

$vcxSrv = 'F:\VcXsrv\vcxsrv.exe'
Get-CimInstance Win32_Process -Filter "Name='vcxsrv.exe'" |
    Where-Object { $_.ExecutablePath -eq $vcxSrv -and $_.CommandLine -match '(^|\s):1(\s|$)' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

Write-Host 'SO-ARM101 MoveIt/RViz and its VcXsrv display have been stopped.' -ForegroundColor Green
