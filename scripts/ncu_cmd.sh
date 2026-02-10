# CUDA_VISIBLE_DEVICES=2 ncu --set full \
#   --graph-profiling graph --target-processes all \
#   -o universal_operators -f \
#   ./build/tests_cpp/cublas/universal_operators/test_universal_operators


CUDA_VISIBLE_DEVICES=2 ncu \
  --kernel-name regex:"silu_kernel_impl" \
  --set full \
  --target-processes all \
  -o universal_operators_silu_impl \
  -f \
  ./build/tests_cpp/cublas/universal_operators/test_universal_operators

CUDA_VISIBLE_DEVICES=2 ncu \
  --kernel-name regex:"silu_kernel_vec" \
  --set full \
  --target-processes all \
  -o universal_operators_silu_vec \
  -f \
  ./build/tests_cpp/cublas/universal_operators/test_universal_operators

CUDA_VISIBLE_DEVICES=2 ncu \
  --kernel-name regex:"silu_kernel_v3" \
  --set full \
  --target-processes all \
  -o universal_operators_silu_v3 \
  -f \
  ./build/tests_cpp/cublas/universal_operators/test_universal_operators


CUDA_VISIBLE_DEVICES=2 ncu \
  --kernel-name regex:"add_bias_kernel_impl" \
  --set full \
  --target-processes all \
  -o universal_operators_bias_impl \
  -f \
  ./build/tests_cpp/cublas/universal_operators/test_universal_operators

CUDA_VISIBLE_DEVICES=2 ncu \
  --kernel-name regex:"add_bias_kernel_vec" \
  --set full \
  --target-processes all \
  -o universal_operators_bias_vec \
  -f \
  ./build/tests_cpp/cublas/universal_operators/test_universal_operators

CUDA_VISIBLE_DEVICES=2 ncu \
  --kernel-name regex:"add_bias_kernel_v3" \
  --set full \
  --target-processes all \
  -o universal_operators_bias_v3 \
  -f \
  ./build/tests_cpp/cublas/universal_operators/test_universal_operators