# How to use

## qdq pattern
qdq only in fwd path for input and only in bwd for output.

## To compare convergence
TF_XLA_FLAGS="--tf_xla_auto_jit=2" XLA_FLAGS="--xla_gpu_enable_cublaslt=true" python tf_mnist.py --fp8

V.S. non-fp8

TF_XLA_FLAGS="--tf_xla_auto_jit=2" XLA_FLAGS="--xla_gpu_enable_cublaslt=true" python tf_mnist.py 


## Issue with bias
TF_XLA_FLAGS="--tf_xla_auto_jit=2" XLA_FLAGS="--xla_gpu_enable_cublaslt=true" python tf_mnist.py --fp8 --bias
gives nan. JAX equivalent doens't have such issue with bias.
