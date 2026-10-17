#!/usr/bin/env bash
# Build Contact-GraspNet's PointNet++ TensorFlow custom ops (sampling, grouping, 3d_interpolation)
# for Jetson Orin + CUDA 13.x + the TensorFlow in the active venv.
#
# Usage (venv active):   bash compile_tf_ops.sh [path/to/contact_graspnet]
# Optional overrides:    CUDA_HOME, CUDA_ARCH (default 87 = Orin), CXX_STD (default c++17), TF_INCLUDE
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
read -r -a TF_CFLAGS <<< "$(tf_query 'print(" ".join(tf.sysconfig.get_compile_flags()))')"
read -r -a TF_LFLAGS <<< "$(tf_query 'print(" ".join(tf.sysconfig.get_link_flags()))')"

[[ -n "$TF_INC" ]] || die "could not import tensorflow with this python ($(command -v python))"
[[ -f "$TF_INC/tensorflow/core/framework/op_kernel.h" ]] \
  || die "TensorFlow C++ headers not found in '$TF_INC'. This wheel may not ship them; point TF_INCLUDE at a TF 2.21 include directory."
compgen -G "$TF_LIB/libtensorflow_framework.so*" >/dev/null \
  || die "libtensorflow_framework.so not found in '$TF_LIB'"

EXTRA_INC=()
if [[ -d "$TF_INC/external/nsync/public" ]]; then EXTRA_INC+=(-I"$TF_INC/external/nsync/public"); fi

echo "CUDA_HOME=$CUDA_HOME  (include: $CUDA_INC, lib: $CUDA_LIB)"
echo "nvcc: $("$NVCC" --version | tail -n1)"
echo "TF include: $TF_INC"
echo "TF flags: ${TF_CFLAGS[*]} | ${TF_LFLAGS[*]}"
echo "Arch: sm_$ARCH   C++ standard: $CXX_STD"

# --- Build --------------------------------------------------------------------------------
built=()
shopt -s nullglob
for dir in "$OPS"/*/; do
  dir="${dir%/}"
  cpps=("$dir"/tf_*.cpp)
  cus=("$dir"/*.cu)
  (( ${#cpps[@]} > 0 )) || continue

  objs=()
  for cu in "${cus[@]}"; do
    obj="${cu%.cu}.cu.o"
    echo "==> nvcc  $(basename "$dir")/$(basename "$cu")"
    "$NVCC" -std="$CXX_STD" -O2 -x cu -Xcompiler -fPIC -DGOOGLE_CUDA=1 \
      -gencode "arch=compute_${ARCH},code=sm_${ARCH}" \
      -gencode "arch=compute_${ARCH},code=compute_${ARCH}" \
      -c "$cu" -o "$obj"
    objs+=("$obj")
  done

  cpp="${cpps[0]}"
  out="$dir/$(basename "$cpp" .cpp)_so.so"
  echo "==> g++   $(basename "$dir")/$(basename "$cpp") -> $(basename "$out")"
  g++ -std="$CXX_STD" -O2 -shared -fPIC \
    -I"$TF_INC" "${TF_CFLAGS[@]}" "${EXTRA_INC[@]}" -I"$CUDA_INC" \
    "$cpp" "${objs[@]}" -o "$out" \
    -L"$CUDA_LIB" -lcudart "${TF_LFLAGS[@]}" -Wl,-rpath,"$CUDA_LIB"
  built+=("$out")
done
shopt -u nullglob

(( ${#built[@]} > 0 )) || die "no tf_*.cpp files found under $OPS"

echo "==> Loading each library in TensorFlow"
python - "${built[@]}" <<'EOF'
import sys
import tensorflow as tf
for path in sys.argv[1:]:
    tf.load_op_library(path)
    print("OK  ", path)
EOF
