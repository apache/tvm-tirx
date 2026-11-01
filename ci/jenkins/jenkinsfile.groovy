// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// Reject untrusted helper edits before allocating nodes or running pipeline steps.
def helper_source
try {
  helper_source = readTrusted('ci/jenkins/common.groovy')
} catch (Throwable ex) {
  echo 'Jenkinsfile changes must be pushed to an upstream branch. ' +
    'Check the branch build when PR helpers differ from the trusted target.'
  throw ex
}
ci = evaluate(helper_source)
ci.start()

parallel(
  'ARM': {
    stage('BUILD: ARM') {
      ci.with_node(instance: 'ARM-GRAVITON3-SPOT', retry_instance: 'ARM-GRAVITON3',
        name: 'BUILD: ARM', folder: 'tvm/build-arm', variables: ['PLATFORM=arm']) {
        ci.docker_init(images.ci_arm)
        withEnv(ci.build_environment()) {
          sh(
            script: """${docker_run} --no-gpu ${images.ci_arm} bash -euo pipefail -c '
              source ./tests/scripts/task_install_build_dependencies.sh
              ./tests/scripts/task_build_arm.sh
            '""",
            label: 'Build ARM wheels and run C++/Rust tests',
          )
        }
        ci.upload_artifacts('arm', '--path "dist/*.whl"')
      }
    }
  },
  'GPU': {
    stage('BUILD: GPU') {
      ci.with_node(instance: 'CPU-SPOT', retry_instance: 'CPU',
        name: 'BUILD: GPU', folder: 'tvm/build-gpu', variables: ['PLATFORM=gpu']) {
        ci.docker_init(images.ci_gpu)
        withEnv(ci.build_environment()) {
          sh(
            script: """${docker_run} --no-gpu ${images.ci_gpu} bash -euo pipefail -c '
              source ./tests/scripts/task_install_build_dependencies.sh
              ./tests/scripts/task_build_gpu.sh
            '""",
            label: 'Build CUDA wheels and run C++ tests',
          )
        }
        ci.upload_artifacts('gpu', '--path "dist/*.whl"')
      }
    }
    // All consumers use this build's exact FFI and CUDA wheels under CPython 3.12.
    parallel(
      'UT: CPU': {
        stage('UT: CPU') {
          ci.with_node(instance: 'CPU-SPOT', retry_instance: 'CPU',
            name: 'UT: CPU', folder: 'tvm/ut-python-cpu',
            variables: ['PLATFORM=cpu', 'TEST_STEP_NAME=UT: CPU']) {
            ci.docker_init(images.ci_cpu)
            sh 'rm -rf dist'
            ci.download_artifacts('gpu')
            sh(
              script: """${docker_run} --no-gpu ${images.ci_cpu} bash -euo pipefail -c '
                source ./tests/scripts/test_task_install_dist_wheels.sh
                ./tests/scripts/task_python_unittest.sh
                ./tests/scripts/task_minimal_examples_cpu.sh
              '""",
              label: 'Run CPU unit tests and examples on the CUDA wheel',
            )
          }
        }
      },
      'UT: GPU': {
        stage('UT: GPU') {
          ci.with_node(instance: 'GPU-SPOT', retry_instance: 'GPU',
            name: 'UT: GPU', folder: 'tvm/ut-python-gpu',
            variables: ['PLATFORM=gpu', 'TEST_STEP_NAME=UT: GPU']) {
            ci.docker_init(images.ci_gpu)
            sh 'rm -rf dist'
            ci.download_artifacts('gpu')
            sh(
              script: """${docker_run} ${images.ci_gpu} bash -euo pipefail -c '
                # CUDA host tests need the driver stub at link time.
                export LIBRARY_PATH="/usr/local/cuda/lib64/stubs\${LIBRARY_PATH:+:\$LIBRARY_PATH}"
                source ./tests/scripts/test_task_install_dist_wheels.sh --group cuda
                ./tests/scripts/task_python_unittest_gpuonly.sh
                ./tests/scripts/task_minimal_examples_cuda.sh
              '""",
              label: 'Run GPU unit tests and examples',
            )
          }
        }
      },
      'DOCS': {
        stage('DOCS') {
          ci.with_node(instance: 'CPU-SPOT', retry_instance: 'CPU',
            name: 'DOCS', folder: 'tvm/tirx-docs', variables: ['PLATFORM=gpu']) {
            ci.docker_init(images.ci_gpu)
            sh 'rm -rf dist'
            ci.download_artifacts('gpu')
            sh(
              script: "${docker_run} --no-gpu ${images.ci_gpu} ./tests/scripts/task_python_docs.sh",
              label: 'Build TIRx API documentation',
            )
            ci.upload_artifacts('docs', '--path docs.tgz')
          }
        }
      },
    )
  },
  'EXTRA': {
    stage('BUILD: EXTRA') {
      ci.with_node(instance: 'CPU-SPOT', retry_instance: 'CPU',
        name: 'BUILD: EXTRA', folder: 'tvm/build-extra', variables: ['PLATFORM=extra']) {
        ci.docker_init(images.ci_extra)
        withEnv(ci.build_environment()) {
          sh(
            script: """${docker_run} --no-gpu ${images.ci_extra} bash -euo pipefail -c '
              source ./tests/scripts/task_install_build_dependencies.sh
              ./tests/scripts/task_build_gpu_other.sh
            '""",
            label: 'Build OpenCL, Vulkan and ROCm wheels without tests',
          )
        }
        ci.upload_artifacts('extra', '--path "dist/*.whl"')
      }
    }
  },
)

// Deployment requires all three branches and every GPU consumer to succeed.
stage('Deploy Docs') {
  if (env.BRANCH_NAME == 'main' && !env.CHANGE_ID && env.DOCS_DEPLOY_ENABLED == 'yes') {
    ci.with_node(instance: 'CPU', name: 'Deploy Docs', folder: 'tvm/deploy-tirx-docs') {
      sh 'rm -f docs.tgz'
      ci.download_artifacts('docs')
      // Keep credential-bearing deployment in the trusted Jenkinsfile.
      sh(
        script: '''
          set -eu
          SOURCE_COMMIT=$(git rev-parse HEAD)
          rm -rf tvm-site
          git clone --branch asf-site --depth=1 https://github.com/apache/tvm-site.git tvm-site
          cd tvm-site
          git log -1 --format='Site base: %H'
          rm -rf tirx
          mkdir tirx
          tar -xzf ../docs.tgz -C tirx
          test -s tirx/index.html
          test "$(cat tirx/commit_hash)" = "$SOURCE_COMMIT"
          git config user.name tvm-bot
          git config user.email 95660001+tvm-bot@users.noreply.github.com
          git add -f -- tirx
          test -z "$(git diff --cached --name-only -- . ':!:tirx')"
          git diff --cached --shortstat
          git diff --cached --name-only | sed -n '1,20p'
          if ! git diff --cached --quiet; then
            git commit --quiet -m "deploying tirx docs ($(cat tirx/commit_hash))" -- tirx
          fi
          git log -1 --format='Docs preview commit: %H'
        ''',
        label: 'Update tvm-site/tirx',
      )
      withCredentials([string(credentialsId: 'docs-push-token', variable: 'GITHUB_TOKEN')]) {
        sh(
          script: '''
            set +x
            set -eu
            cd tvm-site
            auth=$(printf 'x-access-token:%s' "$GITHUB_TOKEN" | base64 | tr -d '\\n')
            for attempt in 1 2 3; do
              if git -c http.extraHeader="Authorization: Basic $auth" push origin HEAD:asf-site; then
                exit 0
              fi
              if [ "$attempt" = 3 ]; then
                exit 1
              fi
              sleep 2
              git pull --rebase origin asf-site
            done
          ''',
          label: 'Publish TIRx documentation to apache/tvm-site',
        )
      }
    }
  }
}
