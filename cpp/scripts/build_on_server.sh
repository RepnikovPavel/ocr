#!/usr/bin/env bash
# Build + test the dots.mocr C++/CUDA engine on server1 over SSH.
#
# 1. rsync the local cpp/ tree to /mnt/nvme2/ocr-flex/cpp on the server.
# 2. Build the docker image dots-mocr-cpp:build once (CUDA 13 devel + cmake).
# 3. cmake configure + build inside a --gpus all container, checkpoint at /ckpt.
# 4. Run the unit tests (tokenizer/loader/imageproc/kernels).
#
# We upload a small remote runner script to dodge nested-quote hell (the docker
# `bash -lc '...'` inside an SSH `bash -lc "..."` is painful to get right).
set -euo pipefail

SERVER=${SERVER:-user@192.168.1.68}
REMOTE_DIR=${REMOTE_DIR:-/mnt/nvme2/ocr-flex}
CKPT=${CKPT:-/mnt/nvme2/dots_mocr_ckpt}
SSH=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 "${SERVER}")
SCP=(scp -o StrictHostKeyChecking=no -o ConnectTimeout=15)
REPO=$(git rev-parse --show-toplevel)

echo ">> rsync cpp/ -> ${SERVER}:${REMOTE_DIR}/cpp/"
rsync -az --rsh="ssh -o StrictHostKeyChecking=no" \
    --exclude=build/ --exclude='*.o' --exclude='__pycache__' \
    "${REPO}/cpp/" "${SERVER}:${REMOTE_DIR}/cpp/"

# Upload the in-container runner (plain, no nested quotes).
cat > /tmp/_dots_remote_build.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${REMOTE_DIR}/cpp
# build/ is created root-owned by the docker runs; wipe it inside a container
# (which runs as root) so the CMake cache + arch flags are fresh each run.
if [ -d build ]; then
  docker run --rm -v \$(pwd):/work/cpp -w /work/cpp alpine rm -rf build || true
fi
if ! docker image inspect dots-mocr-cpp:build >/dev/null 2>&1; then
  echo 'building dots-mocr-cpp:build (one-time, ~2 min)'
  docker build -t dots-mocr-cpp:build -f docker/Dockerfile.build .
fi
echo '>> cmake configure + build'
docker run --rm --gpus all \
  -v \$(pwd):/work/cpp -v ${CKPT}:/ckpt:ro \
  -w /work/cpp dots-mocr-cpp:build \
  bash -lc 'cmake -B build -S . -G Ninja -DCMAKE_CUDA_ARCHITECTURES="89;120" && cmake --build build -j'
echo '>> run unit tests'
docker run --rm --gpus all \
  -v \$(pwd):/work/cpp -v ${CKPT}:/ckpt:ro \
  -w /work/cpp dots-mocr-cpp:build \
  bash -lc './build/test_tokenizer /ckpt/tokenizer.json && ./build/test_loader /ckpt && ./build/test_imageproc && ./build/test_kernels'
echo '>> done'
EOF
"${SCP[@]}" /tmp/_dots_remote_build.sh "${SERVER}:/tmp/_dots_remote_build.sh" >/dev/null
"${SSH[@]}" bash /tmp/_dots_remote_build.sh
