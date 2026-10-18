#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${SIGFIT_IMAGE:-marugoto-sigfit:1.2.0-r3.4.1}"

test "$(git -C "${ROOT}/source/sigfit-v1.2.0" rev-parse HEAD)" = \
  "27a9ff5ff4f491476815b532564ae1e7955ed001"

DOCKER_BUILDKIT=1 BUILDKIT_PROGRESS=plain \
  docker build --pull=false --tag "${IMAGE}" "${ROOT}"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  --volume "${ROOT%/*}:/work" \
  --workdir /work \
  "${IMAGE}" \
  Rscript sigfit_poulos2018_20260726/smoke_test_sigfit_v1.2.0.R

"${ROOT}/run_in_sigfit_v1.2.0.sh" \
  Rscript sigfit_poulos2018_20260726/record_environment.R

