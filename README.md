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

You might wonder: **If my Claude or Codex can just spit out CUDA, PTX, or even SASS directly, why bother with another compiler layer?** We've been asking ourselves the same question.

Our conclusion is that **compilers are still useful, but their role is changing**. Agents can drive the optimization process, while a shared compiler foundation provides reusable ways to represent kernels, analyze their behavior, and generate executable code. That foundation should give agents the freedom to build their own abstractions and transformations as they explore new strategies.

**TIRx is our attempt to build a minimal, stable, and extensible DSL and compiler foundation for that future**, enabling agents to write and optimize kernels, develop their own analyses, and eventually verify correctness.

## Features

- **Minimal, stable core IR** — a small, stable set of core AST nodes, designed for common needs and extensions.
- **Predictable utilities** — TVM-FFI bridge generation, symbolic integer analysis, structural traversal and mapping, and minimal code generation for CUDA and host LLVM.
- **Dynamic extension by design** — agents can build and iterate on extensions in independent packages, without rebuilding a monolithic compiler.
- **First-class cross-language support** — Python, C++, and Rust, allowing agents to prototype rapidly and move performance-critical components across languages.
- **Built on stable toolchains** — leverage established compiler backends instead of reinventing them.

## Documentation

Check out the [documentation](https://tvm.apache.org/tirx/) to get started.
