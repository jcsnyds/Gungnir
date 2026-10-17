#!/usr/bin/env bash
# Build Contact-GraspNet's PointNet++ TensorFlow custom ops (sampling, grouping, 3d_interpolation)
# for Jetson Orin + CUDA 13.x + the TensorFlow in the active venv.
#
# Usage (venv active):   bash compile_tf_ops.sh [path/to/contact_graspnet]
# Optional overrides:    CUDA_HOME, CUDA_ARCH (default 87 = Orin), CXX_STD (default c++17), TF_INCLUDE
#
# Differences from the repo's compile_pointnet_tfops.sh: CUDA/TF paths are discovered instead of
# hard-coded, C++17 (TF 2.21's headers don't compile as C++11), sm_87 code generation, a one-line
# source fix for the removed `Status::OK()`, and all three ops are attempted before reporting.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${1:-$HERE/contact_graspnet}"
OPS="$REPO/pointnet2/tf_ops"
ARCH="${CUDA_ARCH:-87}"
CXX_STD="${CXX_STD:-c++17}"

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -d "$OPS" ]] || die "no tf_ops directory at $OPS (pass the contact_graspnet repo path as the first argument)"
command -v python >/dev/null || die "python not found; activate the venv first"

# --- CUDA ---------------------------------------------------------------------------------
if [[ -z "${CUDA_HOME:-}" ]]; then
  NVCC_PATH="$(command -v nvcc)" || die "nvcc not found on PATH; set CUDA_HOME"
  CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$NVCC_PATH")")")"
fi
NVCC="$CUDA_HOME/bin/nvcc"
[[ -x "$NVCC" ]] || die "no nvcc at $NVCC"

# find_dir <glob> <dir>...: print the first directory that contains a match for <glob>.
find_dir() {
  local pattern="$1"; shift
  local d
  for d in "$@"; do
    if compgen -G "$d/$pattern" >/dev/null; then echo "$d"; return 0; fi
  done
  return 1
}
CUDA_INC="$(find_dir cuda_runtime.h "$CUDA_HOME/include" "$CUDA_HOME"/targets/*/include)" \
  || die "cuda_runtime.h not found under $CUDA_HOME"
CUDA_LIB="$(find_dir 'libcudart.so*' "$CUDA_HOME/lib64" "$CUDA_HOME"/targets/*/lib "$CUDA_HOME/lib")" \
  || die "libcudart not found under $CUDA_HOME"

# --- TensorFlow ---------------------------------------------------------------------------
tf_query() { python -c "import tensorflow as tf; $1" 2>/dev/null | tail -n1 || true; }

TF_INC="${TF_INCLUDE:-$(tf_query 'print(tf.sysconfig.get_include())')}"
TF_LIB="$(tf_query 'print(tf.sysconfig.get_lib())')"
read -r -a TF_CFLAGS_RAW <<< "$(tf_query 'print(" ".join(tf.sysconfig.get_compile_flags()))')"
read -r -a TF_LFLAGS <<< "$(tf_query 'print(" ".join(tf.sysconfig.get_link_flags()))')"

[[ -n "$TF_INC" ]] || die "could not import tensorflow with this python ($(command -v python))"
[[ -f "$TF_INC/tensorflow/core/framework/op_kernel.h" ]] \
  || die "TensorFlow C++ headers not found in '$TF_INC'. This wheel may not ship them; point TF_INCLUDE at a TF 2.21 include directory."
compgen -G "$TF_LIB/libtensorflow_framework.so*" >/dev/null \
  || die "libtensorflow_framework.so not found in '$TF_LIB'"

# tf.sysconfig already adds its own --std=...; drop it so there is exactly one standard (CXX_STD).
TF_CFLAGS=()
for flag in "${TF_CFLAGS_RAW[@]}"; do
  [[ "$flag" == --std=* || "$flag" == -std=* ]] || TF_CFLAGS+=("$flag")
done

EXTRA_INC=()
if [[ -d "$TF_INC/external/nsync/public" ]]; then EXTRA_INC+=(-I"$TF_INC/external/nsync/public"); fi
# TF 2.21's headers are extremely noisy about deprecations; hide those so real errors stay visible.
QUIET=(-Wno-deprecated-declarations -Wno-interference-size -Wno-unused-result)

echo "CUDA_HOME=$CUDA_HOME  (include: $CUDA_INC, lib: $CUDA_LIB)"
echo "nvcc: $("$NVCC" --version | tail -n1)"
echo "TF include: $TF_INC"
echo "TF flags: ${TF_CFLAGS[*]} | ${TF_LFLAGS[*]}"
echo "Arch: sm_$ARCH   C++ standard: $CXX_STD"

# --- Source fix ---------------------------------------------------------------------------
# In TF 2.21 `tensorflow::Status` is just `absl::Status`, which has no static OK(). The ops return
# Status::OK() from their shape functions, so replace it with absl::OkStatus() (backup: *.orig).
patch_status_ok() {
  local f="$1"
  if grep -qE 'Status::OK\(\)' "$f"; then
    echo "==> patching $(basename "$f"): Status::OK() -> absl::OkStatus()  (backup: $(basename "$f").orig)"
    sed -E -i.orig 's/(::)?(tensorflow::)?Status::OK\(\)/absl::OkStatus()/g' "$f"
  fi
}

# --- Build --------------------------------------------------------------------------------
LAST_OUT=""

# build_dir <dir>: compile one op directory; sets LAST_OUT. Every command is guarded with
# `|| return 1` because this runs inside `if`, where `set -e` is disabled.
build_dir() {
  local dir="$1" cu obj cpp
  local -a cpps cus objs
  cpps=("$dir"/tf_*.cpp)
  cus=("$dir"/*.cu)
  objs=()
  cpp="${cpps[0]}"
  patch_status_ok "$cpp"

  for cu in "${cus[@]}"; do
    obj="${cu%.cu}.cu.o"
    echo "==> nvcc  $(basename "$dir")/$(basename "$cu")"
    "$NVCC" -std="$CXX_STD" -O2 -x cu -Xcompiler -fPIC -DGOOGLE_CUDA=1 \
      -gencode "arch=compute_${ARCH},code=sm_${ARCH}" \
      -gencode "arch=compute_${ARCH},code=compute_${ARCH}" \
      -I"$CUDA_INC" -I"$TF_INC" "${TF_CFLAGS[@]}" \
      -c "$cu" -o "$obj" || return 1
    objs+=("$obj")
  done

  LAST_OUT="$dir/$(basename "$cpp" .cpp)_so.so"
  echo "==> g++   $(basename "$dir")/$(basename "$cpp") -> $(basename "$LAST_OUT")"
  g++ -std="$CXX_STD" -O2 -shared -fPIC "${QUIET[@]}" \
    -I"$TF_INC" "${TF_CFLAGS[@]}" "${EXTRA_INC[@]}" -I"$CUDA_INC" \
    "$cpp" "${objs[@]}" -o "$LAST_OUT" \
    -L"$CUDA_LIB" -lcudart "${TF_LFLAGS[@]}" -Wl,-rpath,"$CUDA_LIB" || return 1
}

built=()
failed=()
shopt -s nullglob
for dir in "$OPS"/*/; do
  dir="${dir%/}"
  cpps=("$dir"/tf_*.cpp)
  (( ${#cpps[@]} > 0 )) || continue
  if build_dir "$dir"; then built+=("$LAST_OUT"); else failed+=("$(basename "$dir")"); fi
done
shopt -u nullglob

(( ${#built[@]} + ${#failed[@]} > 0 )) || die "no tf_*.cpp files found under $OPS"
if (( ${#failed[@]} > 0 )); then
  echo >&2
  echo "BUILD FAILED for: ${failed[*]}" >&2
  echo "Built OK: ${built[*]:-none}" >&2
  exit 1
fi

# --- Verify -------------------------------------------------------------------------------
echo "==> Loading each library in TensorFlow"
python - "${built[@]}" <<'EOF'
import sys
import tensorflow as tf
for path in sys.argv[1:]:
    tf.load_op_library(path)
    print("OK  ", path)
EOF

echo "==> Running each op once on the GPU (through the repo's Python wrappers)"
python - "$OPS" <<'EOF' || { echo "The libraries built and loaded, but a test run failed. Paste the traceback above." >&2; exit 1; }
import os, sys
import numpy as np
import tensorflow as tf

ops_dir = sys.argv[1]
for sub in ("sampling", "grouping", "3d_interpolation"):
    sys.path.insert(0, os.path.join(ops_dir, sub))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point
from tf_interpolate import three_nn, three_interpolate

with tf.device("/GPU:0"):
    xyz = tf.constant(np.random.rand(2, 1024, 3).astype(np.float32))
    idx = farthest_point_sample(128, xyz)
    new_xyz = gather_point(xyz, idx)
    print("sampling     ->", new_xyz.shape)
    g_idx, _ = query_ball_point(0.3, 16, xyz, new_xyz)
    grouped = group_point(xyz, g_idx)
    print("grouping     ->", grouped.shape)
    dist, nn_idx = three_nn(xyz, new_xyz)
    feats = tf.constant(np.random.rand(2, 128, 8).astype(np.float32))
    out = three_interpolate(feats, nn_idx, tf.ones_like(dist) / 3.0)
    print("interpolate  ->", out.shape)
print("All three ops ran.")
EOF
