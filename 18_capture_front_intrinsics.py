#!/opt/lerobot061/bin/python
"""Capture intrinsics-only dataset for the FRONT camera (project 1, B).

The front camera is fixed overhead.  Intrinsics only need the checkerboard
seen from many poses -- NO robot motion required.  Hold the board by hand,
fill a large fraction of the frame, vary distance and tilt.

Usage (board):

  cd /root/lerobot_project
  /opt/lerobot061/bin/python 18_capture_front_intrinsics.py --poses 18

Browser preview:  http://192.168.0.2:8091
Terminal keys:    ENTER=capture  r=retake  q=quit

Output: /root/lerobot_project/calib_front_intrinsic/
  sample_000.jpg ...  samples.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from atlas_runner import FRONT_CAMERA_PID, camera_device, environment

os.environ.update(environment())

HTTP_PORT = 8091
STATE = {"jpeg": None, "frame": None, "samples": [], "out": None, "max": 0}


def http_serve() -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/preview.jpg":
                img = STATE["jpeg"]
                self._send(img if img is not None else b"", "image/jpeg")
            elif path == "/capture":
                if STATE.get("cb"):
                    STATE["cb"]()
                self._send(b'{"ok":true}', "application/json")
            elif path == "/retake":
                if STATE.get("rb"):
                    STATE["rb"]()
                self._send(b'{"ok":true}', "application/json")
            elif path == "/quit":
                if STATE.get("qb"):
                    STATE["qb"]()
                self._send(b'{"ok":true}', "application/json")
            elif path == "/samples.json":
                self._send(json.dumps(STATE["samples"]).encode(), "application/json")
            elif path.startswith("/sample/"):
                idx = path.split("/")[2]
                p = STATE["out"] / f"sample_{int(idx):03d}.jpg"
                if p.exists():
                    self._send(p.read_bytes(), "image/jpeg")
                else:
                    self._send(b"not found", "text/plain")
            else:
                self._send(b"not found", "text/plain")

    server = ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), Handler)
    print(f"[HTTP] preview http://192.168.0.2:{HTTP_PORT}")
    server.serve_forever()


HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>front intrinsics</title>
<style>body{background:#111;color:#eee;font-family:system-ui;margin:16px}
img{width:100%;max-width:800px;display:block;background:#000}
button{font-size:16px;padding:10px 22px;margin:6px;border:none;border-radius:6px;cursor:pointer}
#c{background:#2e7d32;color:#fff}#r{background:#b26a00;color:#fff}#q{background:#b71c1c;color:#fff}
.s{display:inline-block;border:1px solid #444;margin:4px;padding:3px;background:#1c1c1c}
.s img{height:60px}</style></head><body>
<h2>前置相机内参标定（手拿棋盘格拍）</h2>
<p id="st">0 / 18</p>
<img id="pv">
<p><button id="c">拍照</button><button id="r">重拍</button><button id="q">结束</button></p>
<div id="s"></div>
<script>
const pv=document.getElementById('pv');
setInterval(()=>{pv.src='/preview.jpg?t='+Date.now();},300);
async function ls(){const a=await (await fetch('/samples.json')).json();
document.getElementById('st').textContent=a.length+' / 18';
document.getElementById('s').innerHTML=a.map(x=>'<span class="s"><img src="/sample/'+x.index+'"></span>').join('');}
setInterval(ls,1000);
c.onclick=()=>fetch('/capture');r.onclick=()=>fetch('/retake');q.onclick=()=>fetch('/quit');
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", type=int, default=18)
    parser.add_argument("--out", default="/root/lerobot_project/calib_front_intrinsic")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    STATE["out"] = out
    STATE["max"] = args.poses

    front = camera_device(FRONT_CAMERA_PID, "front")
    cap = cv2.VideoCapture(front, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"cannot open front camera {front}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    samples: list[dict] = []
    index = 0
    last = 0.0

    def capture() -> None:
        nonlocal index, last
        if time.time() - last < 0.5:
            return
        last = time.time()
        p = out / f"sample_{index:03d}.jpg"
        cv2.imwrite(str(p), STATE["frame"])
        samples.append({"index": index})
        print(f"[{index:03d}] captured", flush=True)
        index += 1

    def retake() -> None:
        nonlocal index, last
        if index == 0 or time.time() - last < 0.5:
            return
        last = time.time()
        index -= 1
        (out / f"sample_{index:03d}.jpg").unlink(missing_ok=True)
        samples.pop()
        print(f"[{index:03d}] retaken", flush=True)

    def quit_all() -> None:
        os._exit(0)

    STATE["cb"], STATE["rb"], STATE["qb"] = capture, retake, quit_all
    threading.Thread(target=http_serve, daemon=True).start()
    print(f"Front intrinsics capture, target {args.poses} poses.")
    print(f"Browser: http://192.168.0.2:{HTTP_PORT}  Terminal: ENTER/r/q")

    def stdin_loop() -> None:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                r, _, _ = select.select([sys.stdin], [], [], 0.2)
                if r:
                    ch = sys.stdin.read(1)
                    if ch in ("\r", "\n"):
                        capture()
                    elif ch == "r":
                        retake()
                    elif ch == "q":
                        quit_all()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    threading.Thread(target=stdin_loop, daemon=True).start()

    global frame  # noqa: PLW0603
    STATE["frame"] = None
    try:
        while index < args.poses:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.2)
                continue
            STATE["frame"] = frame
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                STATE["jpeg"] = buf.tobytes()
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()

    (out / "samples.json").write_text(json.dumps(samples), encoding="utf-8")
    print(f"DONE: {len(samples)} samples in {out}")


if __name__ == "__main__":
    main()
