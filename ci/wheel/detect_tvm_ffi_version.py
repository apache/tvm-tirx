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

"""Detect the source TVM FFI release or prepare the crate's registry dependency.

Run ``python ci/wheel/detect_tvm_ffi_version.py detect`` to print the release version
without changing the checkout. After preparing wheel metadata, run
``python ci/wheel/detect_tvm_ffi_version.py rewrite --version VERSION`` to replace the
crate's source dependency with the exact registry version. GitHub output/env
files are written when their environment variables are set.
"""

import argparse
import os
import re
import subprocess
from pathlib import Path

VERSION_PATTERN = r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?"


def detect_version(root):
    """Resolve the committed submodule to its upstream release tag."""
    submodule = subprocess.check_output(
        ["git", "rev-parse", "HEAD:3rdparty/tvm-ffi"], cwd=root, text=True
    ).strip()
    refs = {}
    for line in subprocess.check_output(
        ["git", "ls-remote", "https://github.com/apache/tvm-ffi", "refs/tags/v*"],
        cwd=root,
        text=True,
    ).splitlines():
        sha, ref = line.split()
        refs[ref] = sha
    # Prefer the peeled commit for annotated tags; lightweight tags point at it directly.
    versions = sorted(
        ref.removeprefix("refs/tags/v")
        for ref, sha in refs.items()
        if re.fullmatch("refs/tags/v" + VERSION_PATTERN, ref)
        and refs.get(ref + "^{}", sha) == submodule
    )
    # A release is tagged on its last candidate's commit; the release wins.
    releases = [version for version in versions if "-" not in version.split("+")[0]]
    if len(releases) == 1:
        return releases[0]
    if len(versions) != 1:
        raise SystemExit(
            f"3rdparty/tvm-ffi at {submodule} must match exactly one apache/tvm-ffi "
            f"release tag; found {versions}"
        )
    return versions[0]


def write_github_values(variable, values):
    """Export values in Actions while keeping local invocations independent of it."""
    if path := os.environ.get(variable):
        with open(path, "a", encoding="utf-8", newline="\n") as output:
            for name, value in values.items():
                output.write(f"{name}={value}\n")


def rewrite_dependency(root, version):
    """Replace the source dependency with the verified exact registry version."""
    import tomlkit

    if not re.fullmatch(VERSION_PATTERN, version):
        raise SystemExit("version must come from the verified submodule release tag")
    main = tomlkit.parse((root / "pyproject.toml").read_text(encoding="utf-8"))
    crate_path = root / "rust/tvm-tirx/Cargo.toml"
    crate = tomlkit.parse(crate_path.read_text(encoding="utf-8"))
    # Remove the source path entirely. An exact requirement preserves prerelease versions
    # such as 0.1.15-post2, which a caret requirement could replace with the older 0.1.15.
    crate["dependencies"]["tvm-ffi"] = f"={version}"
    crate_path.write_text(tomlkit.dumps(crate), encoding="utf-8")
    write_github_values(
        "GITHUB_ENV",
        {
            "RELEASE_VERSION": main["project"]["version"],
            "CRATE_VERSION": crate["package"]["version"],
            "TVM_FFI_VERSION": version,
        },
    )
    print(f"Crate tvm-tirx {crate['package']['version']} with crates.io tvm-ffi {version}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("detect", help="print the release version without modifying sources")
    rewrite = commands.add_parser("rewrite", help="prepare the crate's exact registry dependency")
    rewrite.add_argument("--version", required=True, help="version returned by detect")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "detect":
        version = detect_version(root)
        write_github_values("GITHUB_OUTPUT", {"version": version})
        print(version)
    else:
        rewrite_dependency(root, args.version)


if __name__ == "__main__":
    main()
