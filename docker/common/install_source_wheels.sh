#!/bin/bash
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

INPUT_WHEEL_DIR=$(pwd)/wheels

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --input-wheel-dir)
        INPUT_WHEEL_DIR="$2"
        shift 2
        ;;
    --environment)
        ENVIRONMENT="$2"
        shift 2
        ;;
    *)
        echo "Unknown option: $1"
        echo "Usage: $0 --input-wheel-dir DIR"
        exit 1
        ;;
    esac
done

# Check if required arguments are provided
if [ -z "$INPUT_WHEEL_DIR" ] || [ -z "$ENVIRONMENT" ]; then
    echo "Error: --input-wheel-dir and --environment are required"
    echo "Usage: $0 --input-wheel-dir DIR --environment ENV"
    exit 1
fi

if [ "$ENVIRONMENT" = "dev" ]; then
    TE_WHEEL=$(ls $INPUT_WHEEL_DIR/transformer_engine*.whl) || true
    [ -z "$TE_WHEEL" ] && TE_WHEEL=$(bash docker/common/build_te.sh --output-wheel-dir $INPUT_WHEEL_DIR | tail -n 1)
fi

MAMBA_WHEEL=$(ls $INPUT_WHEEL_DIR/mamba*.whl) || true
[ -z "$MAMBA_WHEEL" ] && MAMBA_WHEEL=$(bash docker/common/build_mamba.sh --output-wheel-dir $INPUT_WHEEL_DIR | tail -n 1)

CAUSALCONV1D_WHEEL=$(ls $INPUT_WHEEL_DIR/causal_conv1d*.whl) || true
[ -z "$CAUSALCONV1D_WHEEL" ] && CAUSALCONV1D_WHEEL=$(bash docker/common/build_causalconv1d.sh --output-wheel-dir $INPUT_WHEEL_DIR | tail -n 1)

# Override deps that are already present in the base image
# only for dev. Mirror transformer_engine when paywalled/unavailable.
if [ "$ENVIRONMENT" = "dev" ]; then
    TE_WHEEL=$(ls $INPUT_WHEEL_DIR/transformer_engine*.whl 2>/dev/null || true)
    if [ -n "$TE_WHEEL" ]; then
        echo "Found transformer_engine wheel: $TE_WHEEL"
        uv pip install --no-cache-dir --no-deps "$TE_WHEEL"
    else
        echo "Mirroring transformer_engine with local PyTorch AMP/bfloat16 fallback"
        uv pip install --no-cache-dir --no-deps torch
    fi
fi

# Install heavy optional deps like mamba, causalconv1d
uv pip install --no-cache-dir \
    $MAMBA_WHEEL \
    $CAUSALCONV1D_WHEEL \
    "setuptools>=80"
