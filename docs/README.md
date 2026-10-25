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

# TIRx Documentation

From the repository root, sync the documentation dependencies and editable
package into the project `.venv`:

```bash
uv sync --no-default-groups --group docs
uv run --no-sync python -m sphinx -b html -W docs docs/_build/html
```

Set `BUILD_CPP_DOCS=1` to also generate the C++ API reference; this requires Doxygen.

Open `docs/_build/html/index.html` to browse the Developer Manual and API reference.
See [Build This Doc Site](dev/doc_build.rst) for local serving and clean rebuilds.
