# fp8 einsum dense layer_test



## Getting started
rm -f /tmp/generated/*


TF_DUMP_GRAPH_PREFIX=/tmp/generated TF_XLA_FLAGS="--tf_xla_clustering_debug --tf_xla_auto_jit=2" XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=/tmp/generated --xla_gpu_enable_cublaslt=true" python test_einsum.py


grep "__cublas$lt$matmul$f8" /tmp/generated/*


## To reproduce cublaslt matmul error
CUBLASLT_LOG_LEVEL=5 XLA_FLAGS="--xla_gpu_enable_cublaslt=true " python test_einsum.py
