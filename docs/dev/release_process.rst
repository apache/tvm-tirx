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

Release Process
===============

This guide is for the TIRx release manager. It follows the source-candidate,
vote, and publication sequence used by `TVM-FFI
<https://tvm.apache.org/ffi/dev/release_process.html>`__, with TIRx's own
version files and wheel workflow. Ordinary development needs only
:doc:`source_build` and :doc:`ci_cd`.

TIRx has a manual wheel-build/publication workflow. It does not currently have
a source-release, signing, voting, or version-bump script. The steps below use
Git, GNU tar (``tar`` on Linux, ``gtar`` on macOS), GPG, ``shasum``, and SVN.
They require the release manager's ASF account and signing key, and publication
access where applicable. Follow the `ASF release policy
<https://www.apache.org/legal/release-policy.html>`__ and coordinate the schedule,
artifact names, and release directory with the TVM PMC.

Step 0. Prepare the Candidate
-----------------------------

Agree on the release scope and notes with the community. Resolve release
blockers and verify CI before choosing a commit. TIRx currently stores its
version in three places; update them together in a reviewed change:

- ``pyproject.toml``: ``project.version`` (distribution metadata).
- ``CMakeLists.txt``: ``TVM_TIRX_VERSION`` (native build).
- ``python/tvm_tirx/libinfo.py``: ``__version__`` (Python package).

Tags do not rewrite these values. A candidate for a final release can carry
the final package version while its Git tag has an ``-rcN`` suffix. The examples
below use an illustrative ``0.1.0`` release; choose the agreed version and a
new candidate number for each attempt:

.. code-block:: bash

   export TIRX_VERSION="0.1.0"
   export TIRX_RC="v${TIRX_VERSION}-rc0"
   export TIRX_TAG="v${TIRX_VERSION}"
   export ASF_USERNAME="your-apache-username"

From the reviewed release commit, create and push the candidate tag:

.. code-block:: bash

   git tag -s "$TIRX_RC" -m "Apache TVM TIRx $TIRX_VERSION release candidate"
   git push upstream "$TIRX_RC"

Here ``upstream`` must name ``https://github.com/apache/tvm-tirx.git`` (or its
SSH equivalent). Record the candidate commit and submodule revisions with the
release notes. Ensure the signing key is available in the project's published
``KEYS`` file before asking others to verify it.

Step 1. Build and Stage Source Artifacts
----------------------------------------

Use a fresh release workspace. Clone the candidate and initialize submodules
at that revision, so the archive contains the matching TVM-FFI sources rather
than empty submodule directories:

.. code-block:: bash

   git clone --branch "$TIRX_RC" --recursive \
     https://github.com/apache/tvm-tirx.git "apache-tvm-tirx-$TIRX_VERSION"
   git -C "apache-tvm-tirx-$TIRX_VERSION" submodule status --recursive
   mkdir "tvm-tirx-$TIRX_RC"
   tar --exclude-vcs -czf \
     "tvm-tirx-$TIRX_RC/apache-tvm-tirx-src-$TIRX_VERSION.tar.gz" \
     "apache-tvm-tirx-$TIRX_VERSION"

On macOS, use ``gtar`` for that archive command. Inspect the archive for
``LICENSE``, ``NOTICE``, third-party license files, complete submodule sources,
and the absence of build outputs or Git metadata. GitHub's automatically
generated source archives do not include recursive submodule contents.
Use the final artifact basename from the start so promotion preserves both
signed bytes and the filename in the checksum file.

Sign and checksum the source archive:

.. code-block:: bash

   (
     cd "tvm-tirx-$TIRX_RC"
     gpg --armor --detach-sign "apache-tvm-tirx-src-$TIRX_VERSION.tar.gz"
     shasum -a 512 "apache-tvm-tirx-src-$TIRX_VERSION.tar.gz" \
       > "apache-tvm-tirx-src-$TIRX_VERSION.tar.gz.sha512"
   )

Stage the candidate under the PMC's development distribution area:

.. code-block:: bash

   svn checkout --depth=files https://dist.apache.org/repos/dist/dev/tvm svn-tvm-dev
   cp -R "tvm-tirx-$TIRX_RC" svn-tvm-dev/
   (
     cd svn-tvm-dev
     svn add "tvm-tirx-$TIRX_RC"
     svn commit --username "$ASF_USERNAME" -m "Stage TIRx $TIRX_RC"
   )

A Python sdist built by ``uv build --sdist`` follows ``pyproject.toml``'s
packaging include list. It is a separate artifact from this complete source
archive, and the current publication workflow uploads wheels only. Do not
substitute an unreviewed sdist or GitHub archive for the voted source artifact.

Step 2. Verify and Vote
-----------------------

Download the staged archive, signature, and checksum into a clean directory.
Verify the published signing key's fingerprint through a trusted channel, then
check the downloaded files:

.. code-block:: bash

   gpg --verify "apache-tvm-tirx-src-$TIRX_VERSION.tar.gz.asc" \
     "apache-tvm-tirx-src-$TIRX_VERSION.tar.gz"
   shasum -a 512 -c "apache-tvm-tirx-src-$TIRX_VERSION.tar.gz.sha512"
   tar -xzf "apache-tvm-tirx-src-$TIRX_VERSION.tar.gz"

Build and test from the extracted source using :doc:`source_build` and
:doc:`ci_cd`, skipping the Git checkout/submodule commands because the sources
are already included. Verify package and native versions, license files,
documentation, and the intended platforms. Test the archive itself; a successful
build from the Git checkout does not prove the archive is complete.

For binary verification, run the `wheel workflow
<https://github.com/apache/tvm-tirx/actions/workflows/publish_wheel.yml>`__ with
``tag=$TIRX_RC`` and ``publish_repository=none``. Check the ``tirx-wheel-*``
artifacts and their isolated tests. Record the run and source commit; these
wheels are convenience binaries built from the candidate, not a substitute
for the source-release vote.

Call a ``[VOTE] Release Apache TVM TIRx <version> (<candidate>)`` thread on
``dev@tvm.apache.org``. Include the source distribution URL, candidate tag and
commit, checksums, signing-key information, release notes, and verification
instructions. A GitHub tracking issue can organize the work; keep the official
vote and its result on the development mailing list. Normally allow at least
72 hours. Passing requires at least three binding PMC +1 votes and more binding
positive than negative votes. Post a ``[RESULT]`` with the tally and vote link.
If fixes are needed, prepare a new commit and candidate; do not overwrite the
artifacts or tag that reviewers already examined.

Step 3. Publish the Approved Release
------------------------------------

After the vote passes, return to the reviewed checkout used in Step 0, where
``upstream`` names the Apache repository. Create the final tag at the **same
commit** as the approved candidate. Keep the candidate tag for traceability:

.. code-block:: bash

   git fetch upstream --tags
   git tag -s "$TIRX_TAG" "${TIRX_RC}^{commit}" -m "Apache TVM TIRx $TIRX_VERSION"
   git push upstream "$TIRX_TAG"

Promote the approved SVN directory without rebuilding or renaming its files:

.. code-block:: bash

   svn copy \
     "https://dist.apache.org/repos/dist/dev/tvm/tvm-tirx-$TIRX_RC" \
     "https://dist.apache.org/repos/dist/release/tvm/tvm-tirx-$TIRX_TAG" \
     --username "$ASF_USERNAME" -m "Release Apache TVM TIRx $TIRX_VERSION"

Verify the public downloads and signatures after distribution propagation.
For wheels, ``publish_wheel.yml`` accepts ``tag=$TIRX_TAG`` and
``publish_repository=pypi``. That path **rebuilds and immediately publishes**
after its matrix succeeds; it cannot promote the earlier run's artifacts.
It records provenance attestations and uses the ``pypi`` environment with
trusted publishing. Maintainers must configure the publication permissions
and environment protection.

Before using that path for an official release, resolve the binary-artifact
procedure with the PMC. ASF policy also requires signing and canonical
distribution of supplied binary packages. This workflow does not produce
detached ASCII signatures or stage wheels in Apache's distribution area;
attestations alone do not complete those steps. Keep ``publish_repository=none``
until a reviewed procedure or tooling can sign and distribute the **exact wheel
bytes** intended for publication. Check those artifacts' source commit, version,
provenance, and tests alongside the approved source release.

Step 4. Follow Up
-----------------

Update the project download page, release notes, and documentation links, then
announce the release to the community. Coordinate site publication with the
website maintainers: this repository's docs script builds ``docs.tgz`` but does
not deploy the site, and TVM-FFI's docs-release workflow is specific to TVM-FFI.
In a separate reviewed change, advance all three version locations to the next
development version. Clean up obsolete candidates in the development
distribution area after verifying the final release; follow ASF retention
policy for older releases.
