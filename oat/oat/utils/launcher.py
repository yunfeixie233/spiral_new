# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import socket


def get_free_port():
    sock = socket.socket()
    sock.bind(("", 0))
    ip, port = sock.getsockname()
    sock.close()
    return port


class DistributedLauncher:
    def __init__(
        self, world_size, rank, local_rank, master_addr, master_port, is_master=False
    ) -> None:
        self._world_size = world_size
        self._rank = rank
        self._local_rank = local_rank
        self._master_addr = master_addr
        self._master_port = master_port
        if is_master:
            self._master_port = self.bind()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        os.environ["LOCAL_RANK"] = str(0)

    def bind(self):
        with socket.socket() as sock:
            sock.bind((self._master_addr, self._master_port))
            return sock.getsockname()[1]
