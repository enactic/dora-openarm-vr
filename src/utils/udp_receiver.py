# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import select
import socket
import threading
import time

class JsonUdpReceiver:
    """Background thread that binds a UDP socket and keeps the latest parsed JSON packet."""

    def __init__(self, host: str, port: int, buf_size: int = 4096) -> None:
        self._host = host
        self._port = port
        self._buf_size = buf_size
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def latest(self) -> dict | None:
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._running = False

    def _parse_packet(self, data: bytes) -> dict | None:
        try:
            line = data.decode("utf-8", errors="replace").strip()
            if not line:
                return None
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    def _loop(self) -> None:
        while self._running:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as srv:
                    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    srv.bind((self._host, self._port))
                    srv.settimeout(1.0)
                    print(f"[receiver] Listening on UDP {self._host}:{self._port}")

                    while self._running:
                        try:
                            data, _ = srv.recvfrom(self._buf_size)
                            last_msg = self._parse_packet(data)

                            # Drain any queued datagrams, keep only the freshest
                            while select.select([srv], [], [], 0.0)[0]:
                                data, _ = srv.recvfrom(self._buf_size)
                                parsed = self._parse_packet(data)
                                if parsed is not None:
                                    last_msg = parsed

                            if last_msg is not None:
                                with self._lock:
                                    self._latest = last_msg

                        except TimeoutError:
                            continue
                        except Exception:
                            pass
            except OSError:
                if self._running:
                    time.sleep(1.0)
