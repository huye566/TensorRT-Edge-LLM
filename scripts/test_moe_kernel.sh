#!/bin/bash

usage() {
    echo "Usage: $0 [--ncu | --nsys | --bare] [other options...]"
    echo "  --ncu    : Use NVIDIA Compute Sanitizer (ncu) for analysis"
    echo "  --nsys   : Use NVIDIA Nsight Systems (nsys) for analysis"
    echo "  --bare   : Execute the program directly (without analysis)"
    echo "  If no options are specified, the program will be executed directly by default."
    exit 1
}

EXEC_MODE="bare"
PROFILER_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --ncu)
            EXEC_MODE="ncu"
            shift
            ;;
        --nsys)
            EXEC_MODE="nsys"
            shift
            ;;
        --bare)
            EXEC_MODE="bare"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            PROFILER_ARGS="$PROFILER_ARGS $1"
            shift
            ;;
    esac
done

export HOME_DIR="$(dirname "$(dirname $(readlink -f $0))")"
echo "HOME_DIR: ${HOME_DIR}"
cd ${HOME_DIR}

. ./scripts/common_config.sh

EXEC_COMMAND="./build/tests_cpp/kernels/moe_kernels_cutlass/test_moe_kernels_cutlass"

case $EXEC_MODE in
    "ncu")
        echo "Running with NVIDIA Compute Sanitizer (ncu)..."
        ncu --set full --graph-profiling graph --target-processes all -o moe_kernel -f $PROFILER_ARGS \
            $EXEC_COMMAND
        ;;
    "nsys")
        echo "Running with NVIDIA Nsight Systems (nsys)..."
        nsys profile -t cudnn,cublas,cuda,nvtx,osrt -s cpu -o moe_kernel --force-overwrite true \
            $PROFILER_ARGS $EXEC_COMMAND
        ;;
    "bare")
        echo "Running bare execution..."
        $EXEC_COMMAND $PROFILER_ARGS
        ;;
    *)
        echo "Unknown execution mode: $EXEC_MODE"
        usage
        ;;
esac

if [ $? -eq 0 ]; then
    echo "Execution completed successfully."
else
    echo "Execution failed with exit code $?."
    exit $?
fi