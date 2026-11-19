#!/usr/bin/env bash
# Build a separate controller so the pinned SONIC checkout stays usable for replay.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
upstream="$root/third_party/GR00T-WholeBodyControl"
patch="$root/scripts/patches/sonic-planner-lockstep.patch"
tools_dir=${SONIC_TOOLS_DIR:-$root/.tools/sonic}
mkdir -p "$tools_dir"
export SONIC_TOOLS_DIR=$(cd "$tools_dir" && pwd)
revision=$(git -C "$root" rev-parse HEAD:third_party/GR00T-WholeBodyControl)
stage=$(mktemp -d "$SONIC_TOOLS_DIR/planner-lockstep.XXXXXX")
trap 'rm -rf -- "$stage"' EXIT

git -C "$upstream" archive "$revision" gear_sonic_deploy | tar -xf - -C "$stage"
patch --batch --directory="$stage" -p1 < "$patch"
export SONIC_BUILD_RPATH="$upstream/gear_sonic_deploy/thirdparty/unitree_sdk2/thirdparty/lib/$(uname -m)"
SONIC_BUILD_TARGET=g1_deploy_onnx_ref bash "$root/scripts/build_sonic_controller.sh" "$stage"
install -m 755 "$stage/gear_sonic_deploy/target/release/g1_deploy_onnx_ref" "$SONIC_TOOLS_DIR/g1_deploy_planner_lockstep"
