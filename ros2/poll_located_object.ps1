# Poll the board's located-object JSON and mirror it to the WSL-visible path.
# Board:   /root/located_object.json   (written by 19_locate_object.py --file)
# Windows: F:\robot_arm_atlas\.tmp\located_object.json
# WSL:     /mnt/f/robot_arm_atlas/.tmp/located_object.json (bridge reads this)
#
# Run in Windows PowerShell (keep open):
#   powershell -ExecutionPolicy Bypass -File F:\robot_arm_atlas\ros2\poll_located_object.ps1

$ErrorActionPreference = 'Stop'
$sshKey  = 'C:\Users\98384\.ssh\codex_robot_arm_ed25519'
$board   = 'root@192.168.0.2'
$remote  = '/root/located_object.json'
$local   = 'F:\robot_arm_atlas\.tmp\located_object.json'
$interval = 0.5

New-Item -ItemType Directory -Force -Path (Split-Path $local) | Out-Null

Write-Host "Polling $board`:$remote every ${interval}s -> $local (Ctrl+C to stop)" -ForegroundColor Green

while ($true) {
    try {
        & scp -q -i $sshKey -o ConnectTimeout=5 -o StrictHostKeyChecking=no $board`:$remote $local 2>$null
    } catch {
        # transient failures (board busy, ssh hiccup) are fine
    }
    Start-Sleep -Milliseconds ($interval * 1000)
}
