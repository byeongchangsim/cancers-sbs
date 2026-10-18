#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${SIGFIT_IMAGE:-marugoto-sigfit:1.2.0-r3.4.1}"

docker image inspect "${IMAGE}" >/dev/null
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --volume "${ROOT}:/work" \
  --workdir /work \
  "${IMAGE}" \
  "$@"

