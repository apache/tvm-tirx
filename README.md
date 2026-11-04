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

**We believe that, in the future, all GPU kernels will be developed, optimized, and verified by AI agents.**

Our vision is to give agents a **minimal, stable, and extensible DSL and compiler foundation** to write and optimize kernels, build the analyses they need, and ultimately verify correctness. That's what we're building with TIRx.

But wait, **why would agents need DSLs and compilers at all?** We've been asking ourselves the same question.

**TL;DR:** Agents still need a way to represent kernels, reason about their behavior, and turn them into executable code. A shared DSL and compiler foundation provides reusable IR, analyses, and code generation, so agents can focus on exploring optimization strategies without rebuilding that infrastructure. Keeping this foundation minimal and extensible lets agents add the abstractions and transformations they need. Read our [blog post](#) for our take.

## Features

- **Minimal, stable core IR** — a small, stable set of core AST nodes, designed for common needs and extensions.
- **Predictable utilities** — TVM-FFI bridge generation, symbolic integer analysis, structural traversal and mapping, and minimal code generation for CUDA and host LLVM.
- **Dynamic extension by design** — agents can build and iterate on extensions in independent packages, without rebuilding a monolithic compiler.
- **First-class cross-language support** — Python, C++, and Rust, allowing agents to prototype rapidly and move performance-critical components across languages.
- **Built on stable toolchains** — leverage established compiler backends instead of reinventing them.

## Documentation

Check out the [documentation](https://tvm.apache.org/tirx/) to get started.
