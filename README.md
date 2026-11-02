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

# TVM TIRx: ML Compiler Foundation for the Agentic Era

📚 [Documentation](https://tvm.apache.org/tirx/) | 🚀 [Quick Start](https://tvm.apache.org/tirx/get_started/quick_start.html)

Apache TVM TIRx is a minimal, open ML compiler foundation for the agentic era.

## Features

- **Minimal, stable core IR** - a small, stable set of core AST nodes, designed for common needs and for extension.
- **Predictable utilities** - TVM-FFI bridge generation, symbolic integer analysis, structural traversal and mapping, and minimal code generation for CUDA and host LLVM.
- **Dynamic extension by design** - all compiler APIs are open, so developers and agents can iterate quickly on extensions in separate packages that link to TIRx dynamically, without rebuilding one monolithic bundle.
- **First-class cross-language support** - Python, C++, and native Rust. Prototype an analysis or extension in Python, then bake it into Rust or C++; extensions work across all supported languages.
- **Built on stable toolchains** - a minimal layer for ML compilation needs that connects to stable compiler toolchains and leverages their backend capabilities when possible.

## Documentation

Check out the [documentation site](https://tvm.apache.org/tirx/) for the full documentation, including:

- [Quick Start](https://tvm.apache.org/tirx/get_started/quick_start.html)
- [IR Guides](https://tvm.apache.org/tirx/ir_guides/overview.html)
- [Programming Guides](https://tvm.apache.org/tirx/programming_guides/index.html)
- [Extending TIRx](https://tvm.apache.org/tirx/advanced/extending_ir_in_rust.html)
