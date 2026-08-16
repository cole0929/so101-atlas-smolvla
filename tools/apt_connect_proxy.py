"""Small HTTP/HTTPS CONNECT proxy for provisioning the USB-attached Atlas board.

It intentionally binds to the Windows side of the board-only USB subnet and
uses only the Python standard library. This is a temporary provisioning aid,
not a general-purpose forward proxy.
"""

from __future__ import annotations

import argparse
import select
import socket
import socketserver
from urllib.parse import urlsplit


MAX_HEADER = 64 * 1024


def relay(left: socket.socket, right: socket.socket) -> None:
    sockets = [left, right]
    while sockets:
        readable, _, exceptional = select.select(sockets, [], sockets, 60)
        if exceptional or not readable:
            return
        for source in readable:
            target = right if source is left else left
            data = source.recv(64 * 1024)
            if not data:
                return
            target.sendall(data)


class ProxyHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        self.request.settimeout(30)
        header = bytearray()
        while b"\r\n\r\n" not in header:
            chunk = self.request.recv(8192)
            if not chunk:
                return
            header.extend(chunk)
            if len(header) > MAX_HEADER:
                return

        head, remainder = bytes(header).split(b"\r\n\r\n", 1)
        lines = head.split(b"\r\n")
        method, target, version = lines[0].decode("latin-1").split(" ", 2)

        if method.upper() == "CONNECT":
            host, separator, port_text = target.rpartition(":")
            if not separator:
                host, port_text = target, "443"
            upstream = socket.create_connection((host.strip("[]"), int(port_text)), 30)
            try:
                self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                if remainder:
                    upstream.sendall(remainder)
                relay(self.request, upstream)
            finally:
                upstream.close()
            return

        parsed = urlsplit(target)
        host = parsed.hostname
        if not host:
            return
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query

        filtered_headers = [
            line
            for line in lines[1:]
            if not line.lower().startswith((b"proxy-connection:", b"connection:"))
        ]
        request_head = (
            f"{method} {path} {version}\r\n".encode("latin-1")
            + b"\r\n".join(filtered_headers)
            + b"\r\nConnection: close\r\n\r\n"
        )
        upstream = socket.create_connection((host, port), 30)
        try:
            upstream.sendall(request_head + remainder)
            relay(self.request, upstream)
        finally:
            upstream.close()


class ThreadingProxy(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    with ThreadingProxy((args.bind, args.port), ProxyHandler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
