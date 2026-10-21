..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

Build This Doc Site
===================

The Sphinx site imports TIRx to generate its Python API reference. Complete
:doc:`source_build` first, including the editable install and import check.
Use the same active mamba environment and repository-local ``.venv`` for these
commands; a GPU is not required to build the documentation.

Build and View
--------------

Install the documentation dependency group and build HTML with warnings treated
as errors:

.. code-block:: bash

   uv pip install --group docs
   uv run --no-sync python -m sphinx -b html -W docs docs/_build/html

Open ``docs/_build/html/index.html``, or serve the output locally:

.. code-block:: bash

   uv run --no-sync python -m http.server --bind 127.0.0.1 \
     --directory docs/_build/html 8000

Visit http://127.0.0.1:8000. Re-run the Sphinx command after edits, then refresh
the browser; the HTTP server does not rebuild pages. Rebuild the editable
package first if you changed native code.

For a full Sphinx rebuild that reports all warnings in one pass:

.. code-block:: bash

   uv run --no-sync python -m sphinx -E -a -b html -W --keep-going \
     docs docs/_build/html

Sphinx generates API stubs in ``docs/reference/python/generated``. If you remove
or rename API pages, delete that generated directory and ``docs/_build`` before
rebuilding to avoid stale output. Keep hand-written reference pages intact.

CI Documentation Build
----------------------

The GPU Jenkins pipeline runs ``tests/scripts/task_python_docs.sh`` after
building wheels. That script installs the wheels from ``dist/`` and the docs
dependencies under ``.local/python``, points ``TVM_TIRX_LIBRARY_PATH`` at those
libraries, and performs a clean warnings-as-errors build. It writes the source
commit to ``docs/_build/html/commit_hash`` and packages the site as ``docs.tgz``.
See :doc:`ci_cd` for the CI images and wheel-build prerequisites.
