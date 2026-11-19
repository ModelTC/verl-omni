# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import errno
import random
import socket
from collections.abc import Sequence

# Linux abstract Unix sockets reserve candidates across rollout processes in
# the same network namespace without occupying the eventual TCPStore socket.
# Keep reservations until this server exits; the kernel also releases them on
# abnormal exit, so there are no stale lock files to clean up.
_PORT_RESERVATIONS: list[socket.socket] = []


def ephemeral_port_range() -> tuple[int, int]:
    try:
        with open("/proc/sys/net/ipv4/ip_local_port_range") as f:
            lo, hi = f.read().split()
        return int(lo), int(hi)
    except (OSError, ValueError):
        return 32768, 60999  # Linux default


def get_non_ephemeral_free_port(address: str = "127.0.0.1") -> int:
    """Pick and reserve a free TCP port for this Linux rollout server.

    The consumer binds the port only after worker spawn + model load, and ports
    inside the ephemeral range can be claimed as source ports by unrelated
    connections in that window (later failing to listen with ``EADDRINUSE``).
    A separate abstract socket prevents cooperating servers from selecting the
    same candidate before TCP bind. Unrelated programs do not honor this lease.
    """
    lo, _ = ephemeral_port_range()
    if lo <= 1024:
        raise RuntimeError(f"Ephemeral port range starts at {lo}; no non-privileged candidate ports below it.")
    candidates = range(1024, lo)
    # Training replicas share seeded PRNG state. Port allocation must not use
    # or advance it (seed=42 previously made every replica choose port 21976).
    start = random.SystemRandom().randrange(len(candidates))
    for offset in range(len(candidates)):
        port = candidates[(start + offset) % len(candidates)]
        reservation = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            try:
                reservation.bind(f"\0verl-omni-master-port-{port}")
            except OSError as exc:
                if exc.errno != errno.EADDRINUSE:
                    raise
                continue
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind((address, port))
                except OSError:
                    continue
            _PORT_RESERVATIONS.append(reservation)
            reservation = None
            return port
        finally:
            if reservation is not None:
                reservation.close()
    raise RuntimeError(f"No free non-ephemeral port on {address} below {lo}.")


def worker_group_port_ranges(master_port_range: Sequence[int] | None, num_groups: int) -> list[list[int] | None]:
    """Slice a rendezvous port range into one disjoint sub-range per worker group.

    Ports are only bound at ``init_model``, after every group has been spawned, so groups
    sharing a range would all pick its first free port.
    """
    if master_port_range is None:
        return [None] * num_groups
    lo, hi = (int(port) for port in master_port_range)
    stride = (hi - lo) // num_groups
    if stride < 1:
        raise ValueError(
            f"trainer.ray_master_port_range={master_port_range} has fewer ports than worker groups ({num_groups})."
        )
    return [[lo + i * stride, hi if i == num_groups - 1 else lo + (i + 1) * stride] for i in range(num_groups)]
