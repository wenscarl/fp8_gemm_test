# fp8_gemm_test



## Getting started
rm -f /tmp/generated/*


TF_DUMP_GRAPH_PREFIX=/tmp/generated TF_XLA_FLAGS="--tf_xla_clustering_debug --tf_xla_auto_jit=2" XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=/tmp/generated --xla_gpu_enable_cublaslt=true" python fp8_gemm_test.py einsum


grep "__cublas$lt$matmul$f8" /tmp/generated/*
