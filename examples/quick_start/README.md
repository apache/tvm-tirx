<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Quick Start Examples

This directory contains all the source code for the
[Quick Start](https://tvm.apache.org/tirx/get_started/quick_start.html) tutorial.

| File | What it does |
| --- | --- |
| `add_one_cpu.py`, `add_one_cuda.py` | Write, compile, and run `add_one`; export it to `add_one_cpu.so` / `add_one_cuda.so` |
| `load_add_one_cpu.py`, `load_add_one_cuda.py` | Load the exported library in a new Python process and run it |
| `inspect_and_transform.py` | Inspect the program and specialize it to 128 elements |
| `load_cpp.cc`, `CMakeLists.txt` | Load `add_one_cpu.so` from C++ using only the runtime libraries |

## Run everything

```bash
pip install apache-tvm-tirx torch
bash run_all_cpu.sh        # on Windows: run_all_cpu.bat (Python steps only)

# needs an NVIDIA GPU with CUDA 13 or newer
pip install "apache-tvm-tirx[cuda]"
bash run_all_cuda.sh
```

## Run step by step

Run the scripts from this directory:

```bash
python add_one_cpu.py
python load_add_one_cpu.py
python inspect_and_transform.py
```

For CUDA:

```bash
python add_one_cuda.py
python load_add_one_cuda.py
```

For C++, run `add_one_cpu.py` first, then build and run the loader with CMake:

```bash
cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo
build/load_cpp
```

or directly with `g++`, using `tvm-tirx-config` for the flags:

```bash
g++ -O2 load_cpp.cc \
    $(tvm-tirx-config --cxxflags --ldflags --libs runtime) \
    -Wl,-rpath,$(tvm-tirx-config --libdir) \
    -Wl,-rpath,$(tvm-ffi-config --libdir) \
    -o load_cpp
./load_cpp
```
