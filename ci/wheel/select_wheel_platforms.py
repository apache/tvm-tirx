# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Select complete publication matrices or a non-publishing platform check."""

import json
import os

CUDA_PLATFORMS = [
    {
        "name": "CUDA runtime sidecar (Linux x86_64, manylinux_2_28)",
        "os": "ubuntu-latest",
        "container": "quay.io/manylinux_cuda/manylinux_2_28_x86_64_cuda13_1:latest",
        "arch": "x86_64",
        "script": "ci/wheel/manylinux_build_libtvm_runtime_cuda.sh",
        "lib": "build-wheel-cuda/lib/libtvm_runtime_cuda.so",
    },
    {
        "name": "CUDA runtime sidecar (Linux aarch64, manylinux_2_28)",
        "os": "ubuntu-24.04-arm",
        "container": "quay.io/manylinux_cuda/manylinux_2_28_aarch64_cuda13_1:latest",
        "arch": "aarch64",
        "script": "ci/wheel/manylinux_build_libtvm_runtime_cuda.sh",
        "lib": "build-wheel-cuda/lib/libtvm_runtime_cuda.so",
    },
    {
        "name": "CUDA runtime sidecar (Windows AMD64)",
        "os": "windows-2022",
        "container": "",
        "arch": "AMD64",
        "script": "ci/wheel/windows_build_libtvm_runtime_cuda.bat",
        "lib": "build-wheel-cuda/lib/tvm_runtime_cuda.dll",
    },
    {
        "name": "CUDA runtime sidecar (Windows ARM64)",
        "os": "windows-11-arm",
        "container": "",
        "arch": "ARM64",
        "script": "ci/wheel/windows_build_libtvm_runtime_cuda.bat",
        "lib": "build-wheel-cuda/lib/tvm_runtime_cuda.dll",
    },
]

WHEEL_PLATFORMS = [
    {
        "name": "Linux x86_64 wheel with CUDA runtime (manylinux_2_28)",
        "os": "ubuntu-latest",
        "arch": "x86_64",
        "build": "cp310-manylinux_x86_64",
        "include_cuda_runtime": "1",
        "artifact_suffix": "linux-x86_64-manylinux_2_28",
    },
    {
        "name": "Linux aarch64 wheel with CUDA runtime (manylinux_2_28)",
        "os": "ubuntu-24.04-arm",
        "arch": "aarch64",
        "build": "cp310-manylinux_aarch64",
        "include_cuda_runtime": "1",
        "artifact_suffix": "linux-aarch64-manylinux_2_28",
    },
    {
        "name": "macOS arm64 wheel with Metal",
        "os": "macos-14",
        "arch": "arm64",
        "build": "cp310-macosx_arm64",
        "include_cuda_runtime": "0",
        "artifact_suffix": "macos-arm64",
        "cmake_defines": "-DTVM_TIRX_USE_METAL=ON",
    },
    {
        "name": "Windows AMD64 wheel with CUDA runtime",
        "os": "windows-2022",
        "arch": "AMD64",
        "build": "cp310-win_amd64",
        "include_cuda_runtime": "1",
        "artifact_suffix": "windows-amd64",
    },
    {
        "name": "Windows ARM64 wheel with CUDA runtime",
        "os": "windows-11-arm",
        "arch": "ARM64",
        "build": "cp312-win_arm64",
        "include_cuda_runtime": "1",
        "artifact_suffix": "windows-arm64",
    },
]


def main():
    platform = os.environ.get("WHEEL_PLATFORM", "all")
    if platform not in ("all", "windows-arm64"):
        raise ValueError(f"Unsupported wheel platform: {platform}")
    if platform != "all" and os.environ.get("PUBLISH_REPOSITORY", "none") != "none":
        raise ValueError("Publishing requires the complete platform matrix")
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        for name, rows in (("cuda", CUDA_PLATFORMS), ("wheels", WHEEL_PLATFORMS)):
            selected = (
                rows if platform == "all" else [row for row in rows if row["arch"] == "ARM64"]
            )
            output.write(f"{name}={json.dumps({'include': selected})}\n")


if __name__ == "__main__":
    main()
