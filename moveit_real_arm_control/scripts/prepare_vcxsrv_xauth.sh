#!/usr/bin/env bash
set -euo pipefail

AUTH_FILE=/mnt/f/robot_arm_atlas/.x11/vcxsrv.Xauthority
DISPLAY_NUMBER=1
WINDOWS_HOST="$(ip route show default | awk '{print $3; exit}')"
COOKIE="$(mcookie)"

mkdir -p "$(dirname "$AUTH_FILE")"
rm -f "$AUTH_FILE"
touch "$AUTH_FILE"
chmod 600 "$AUTH_FILE"

# The X server reads this file, while the client selects the same cookie by
# hostname/display. Add both the WSL-side TCP address and local aliases.
xauth -f "$AUTH_FILE" add "${WINDOWS_HOST}:${DISPLAY_NUMBER}" MIT-MAGIC-COOKIE-1 "$COOKIE"
xauth -f "$AUTH_FILE" add "localhost:${DISPLAY_NUMBER}" MIT-MAGIC-COOKIE-1 "$COOKIE"
xauth -f "$AUTH_FILE" add "$(hostname)/unix:${DISPLAY_NUMBER}" MIT-MAGIC-COOKIE-1 "$COOKIE"

printf '%s\n' "$WINDOWS_HOST"
