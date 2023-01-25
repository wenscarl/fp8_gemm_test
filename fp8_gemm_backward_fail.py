import tensorflow as tf
from tensorflow.python.framework import dtypes

M,N,K=64,32,128

transa, transb= False, True

if not transa and not transb:
  input_shape = [M, K]
  kernel_shape = [K, N]
elif transa and transb:
  input_shape = [K, M]  
  kernel_shape = [N, K]
elif transa and not transb:
  input_shape = [K, M]  
  kernel_shape = [K, N]
else:
  input_shape = [M, K]  
  kernel_shape = [N, K]

inputs = tf.random.normal(shape=input_shape)
kernel = tf.random.normal(shape=kernel_shape)

FAKE_E4M3 = dtypes.float8_e4m3fn
FAKE_E5M2 = dtypes.float8_e5m2

E4M3_MAX = 448.
E5M2_MAX = 57344.
AMAX_HIS_LEN = 16

def get_fp8_max(fake_dtype):
  if fake_dtype == FAKE_E4M3:
    return E4M3_MAX
  else:
    assert fake_dtype == FAKE_E5M2
    return E5M2_MAX

def quantize(x, quantized_dtype, scale):
  dtype_max = get_fp8_max(quantized_dtype)
  scaled_x = tf.clip_by_value(x / scale, -dtype_max, dtype_max)
  return tf.cast(scaled_x, quantized_dtype)

def dequantize(x, wide_dtype, scale):
  return tf.cast(x, wide_dtype) * scale

def quantize_dequantize(x, quantized_dtype, scale):
  orig_dtype = x.dtype
  qx = quantize(x, quantized_dtype, scale)
  return dequantize(qx, orig_dtype, scale)

def update_scale(x, quantized_dtype, scale_var):
  dtype_max = get_fp8_max(quantized_dtype)
  amax = tf.cast(tf.math.reduce_max(tf.math.abs(x)), scale_var.dtype)
  amax = tf.maximum(amax, 2 ** -10)
#  scale_var.assign(1.1 * amax / dtype_max)

def qdq_and_update(x, dtype, scale_var):
  qx = quantize_dequantize(x, dtype, scale_var)
  update_scale(x, dtype, scale_var)
  return qx

def in_qdq(input):
  """Quantize-dequantize both the input and the input's gradient."""
  input_scale=tf.constant(2.0)
  input_grad_scale=tf.constant(0.5)
  qin = qdq_and_update(input, FAKE_E4M3, input_scale)
  return qin

def ker_qdq(input):
  """Quantize-dequantize both the input and the input's gradient."""
  ker_scale=tf.constant(3.0)
  qker = qdq_and_update(input, FAKE_E4M3, ker_scale)
  return qker

@tf.function
def test_me(a, b):
  a = in_qdq(a)
  b = ker_qdq(b)
  return tf.matmul(a,b, transa, transb)
y = test_me(inputs, kernel)
print(y)
