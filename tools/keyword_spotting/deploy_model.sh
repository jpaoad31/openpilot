#!/usr/bin/env bash
# deploy_model.sh — copy the trained keyword-spotting model to a Comma 4 device
#
# Usage:
#   bash tools/keyword_spotting/deploy_model.sh <device-ip>
#
# The device IP is shown in the comma 4 settings screen, or use the
# default USB-tethered address: 192.168.43.1
#
# Prerequisites:
#   - Run prepare_model.py first to produce the .onnx and .npz files
#   - SSH access must be set up (comma devices accept SSH on port 22)

set -e

DEVICE_IP="${1:-192.168.43.1}"
DEVICE_USER="comma"
DEVICE_OPENPILOT="/data/openpilot"
MODEL_DIR="selfdrive/modeld/models"

ONNX_FILE="${MODEL_DIR}/keyword_spotting.onnx"
NPZ_FILE="${MODEL_DIR}/keyword_spotting.npz"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "Deploying keyword spotting model to ${DEVICE_USER}@${DEVICE_IP} ..."

# Verify model files exist locally
if [[ ! -f "${REPO_ROOT}/${ONNX_FILE}" ]]; then
  echo "ERROR: ${ONNX_FILE} not found."
  echo "Run this first:  python tools/keyword_spotting/prepare_model.py"
  exit 1
fi

# Ensure target directory exists on device
ssh "${DEVICE_USER}@${DEVICE_IP}" "mkdir -p ${DEVICE_OPENPILOT}/${MODEL_DIR}"

# Copy model files
scp "${REPO_ROOT}/${ONNX_FILE}" "${DEVICE_USER}@${DEVICE_IP}:${DEVICE_OPENPILOT}/${ONNX_FILE}"
scp "${REPO_ROOT}/${NPZ_FILE}"  "${DEVICE_USER}@${DEVICE_IP}:${DEVICE_OPENPILOT}/${NPZ_FILE}"

echo ""
echo "Model files deployed."
echo ""
echo "Now SSH in and install onnxruntime if not already present:"
echo "  ssh ${DEVICE_USER}@${DEVICE_IP}"
echo "  cd ${DEVICE_OPENPILOT} && uv sync"
echo ""
echo "Then trigger a popup to test:"
echo "  touch /tmp/hazard_trigger   # on device"
echo "  # say 'yes' or 'no' — the popup should dismiss"
