from typing import (Any, Callable, Iterable, List, Mapping, Optional, Sequence,
                    Tuple, Union)
import tensorflow as tf
import flax
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import lax

# Didn't found api
E4M3_MAX = 448
E5M2_MAX = 57344

# Type annotations
Array = jnp.ndarray
Dtype = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]
ActivationFn = Callable[..., Array]

def get_fp8_max(fp8_dtype):
  assert fp8_dtype in (jnp.float8_e4m3fn, jnp.float8_e5m2)
  if fp8_dtype == jnp.float8_e4m3fn:
    return E4M3_MAX
  return E5M2_MAX


def quantize(x, quantized_dtype, scale):
  dtype_max = get_fp8_max(quantized_dtype)
  scaled_x = jnp.clip(x / scale, -dtype_max, dtype_max)
  return scaled_x.astype(quantized_dtype)

def dequantize(x, wide_dtype, scale):
  return x.astype(wide_dtype) * scale

def quantize_dequantize(x, quantized_dtype, scale):
  orig_dtype = x.dtype
  qx = quantize(x, quantized_dtype, scale)
  return dequantize(qx, orig_dtype, scale)

def compute_new_scale(x, quantized_dtype, scale):
  dtype_max = get_fp8_max(quantized_dtype)
  amax = jnp.max(jnp.abs(x)).astype(scale.dtype)
  # Ensure scale != 0 and avoid divide-by-zero.
  amax = jnp.maximum(amax, 2**-10)
  return 1.1 * amax / dtype_max

def qdq_and_new_scale(x, dtype, scale):
  qx = quantize_dequantize(x, dtype, scale)
  new_scale = compute_new_scale(x, dtype, scale)
  return qx, new_scale

def compute_scale(amax, scale, fp8_max, margin=0):
  """Default function to convert amax to scaling factor."""
  exp = jnp.floor(jnp.log2(fp8_max / amax)) - margin
  sf = jnp.round(jax.lax.pow(2., jnp.abs(exp)))
  sf = jnp.where(amax > 0.0, sf, scale)
  sf = jnp.where(jax.lax.is_finite(amax), sf, scale)
  sf = jnp.where(exp < 0, 1.0 / sf, sf)
  # The scaling factor we need equals to the notion of "scale_inv" in
  # TransformerEngine. So, we convert the sf to its reciprocal.
  return 1.0 / sf


def compute_new_scale_and_amax_history(x, quantized_dtype, scale, amax_history):
  dtype_max = get_fp8_max(quantized_dtype)
  amax = jnp.max(jnp.abs(x)).astype(scale.dtype)
  new_amax_history = amax_history
  new_amax_history = jnp.roll(new_amax_history, shift=-1, axis=0)
  new_amax_history = new_amax_history.at[0].set(amax)

  amax_from_history = jnp.max(new_amax_history, axis=0)
  scale_update = compute_scale(amax_from_history, scale, dtype_max)
  return scale_update, new_amax_history

def qdq_and_update_sp(x, dtype, scale, amax_history):
  qx = quantize_dequantize(x, dtype, scale)
  new_scale, new_amax_history = compute_new_scale_and_amax_history(
      x, dtype, scale, amax_history)
  return qx, new_scale, new_amax_history


@jax.custom_vjp
def in_qdq(inp, inp_scale, inp_amax_history):
  qin, new_inp_scale, new_inp_amax_history = qdq_and_update_sp(
      inp, jnp.float8_e4m3fn, inp_scale, inp_amax_history)
  # out_grad_scale is needed in vjp
  return qin, new_inp_scale, new_inp_amax_history

def inp_qdq_fwd(inp, inp_scale, inp_amax_history):
  # new_inp_grad_scale is a dummy value
  qin, new_inp_scale, new_inp_amax_history = in_qdq(
      inp, inp_scale, inp_amax_history)
  return (qin, new_inp_scale, new_inp_amax_history), (inp_scale, inp_amax_history)

def inp_qdq_bwd(res, g):
  inp_grad_scale, inp_amax_history = res
  qin_g, inp_scale_g, inp_amax_history_g = g
  return qin_g, inp_scale_g, inp_amax_history_g


in_qdq.defvjp(inp_qdq_fwd, inp_qdq_bwd)

@jax.custom_vjp
def out_qdq(out, out_grad_scale, out_grad_amax_history, dummy1, dummy2):  # only bwd
  return out, out_grad_scale, out_grad_amax_history

def out_qdq_fwd(out, out_grad_scale, out_grad_amax_history, dummy1, dummy2):
  # new_out_grad_scale and new_out_grad_amax_history is dummy values
  qout, new_out_grad_scale, new_out_grad_amax_history = out_qdq(
      out, out_grad_scale, out_grad_amax_history, dummy1, dummy2)
  return (qout, new_out_grad_scale, new_out_grad_amax_history), (out_grad_scale, out_grad_amax_history, )

def out_qdq_bwd(res, g):
  out_grad_scale, out_grad_amax_history, = res
  qout_g, out_grad_scale_g, out_grad_amax_history_g = g
  out_grad, new_out_grad_scale, new_out_grad_amax_history = qdq_and_update_sp(
      qout_g, jnp.float8_e5m2, out_grad_scale, out_grad_amax_history)
  return out_grad, jnp.zeros_like(out_grad_scale_g), jnp.zeros_like(
      out_grad_amax_history_g), new_out_grad_scale, new_out_grad_amax_history

out_qdq.defvjp(out_qdq_fwd, out_qdq_bwd)

class Dense(nn.Dense):
  features: int
  param_dtype: Dtype = jnp.float32
  kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.lecun_normal()
  amax_history_length: int = 16
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
  activation: Optional[ActivationFn] = None

  @nn.compact
  def __call__(self, inputs):
    kernel = self.param('kernel', self.kernel_init,
                        (inputs.shape[-1], self.features), self.param_dtype)
    bias = self.param(
        'bias', self.bias_init, (self.features,),
        self.param_dtype)

    scale_args = (
        nn.initializers.ones_init(),
        jax.random.PRNGKey(32),
        (1, 1),
        jnp.float32)
    amax_history_args = (
        nn.initializers.zeros_init(),
        jax.random.PRNGKey(32),
        (self.amax_history_length, 1),
        jnp.float32)
    kernel_scale = self.variable('qscale', 'kernel_scale', *scale_args)
    kernel_amax_history = self.variable(
        'qscale', 'kernel_amax_history', *amax_history_args)
    ## share the same function
    kernel, new_kernel_scale, new_kernel_amax_history = in_qdq(
        kernel, kernel_scale.value, kernel_amax_history.value)
    kernel_scale.value = new_kernel_scale
    kernel_amax_history.value = new_kernel_amax_history

    input_scale = self.variable('qscale', 'input_scale', *scale_args)
    input_amax_history = self.variable(
        'qscale', 'input_amax_history', *amax_history_args)
    inputs, new_input_scale, new_input_amax_history = in_qdq(
        inputs, input_scale.value, input_amax_history.value)
    input_scale.value = new_input_scale
    input_amax_history.value = new_input_amax_history

    # Actual dense layer math.
    original_shape = inputs.shape
    assert len(original_shape) >= 2
    a = jnp.reshape(inputs, (-1, original_shape[-1]))
    out = jax.lax.dot(a, kernel)

    output_grad_scale = self.variable(
        'qscale', 'output_grad_scale', *scale_args)
    # output_grad_scale is updated in training loop
    output_grad_scale_perturb = self.variable(
        'grad_qscale_placeholder', 'output_grad_scale_placeholder', *scale_args)

    output_grad_amax_history = self.variable(
        'qscale', 'output_grad_amax_history', *amax_history_args)
    output_grad_amax_history_perturb = self.variable(
        'grad_qscale_placeholder', 'output_grad_amax_history_placeholder', *
        amax_history_args)

    out, new_out_grad_scale, new_out_grad_amax_history = out_qdq(
        out, output_grad_scale.value,
        output_grad_amax_history.value,
        output_grad_scale_perturb.value,
        output_grad_amax_history_perturb.value
    )
    if self.use_bias:
      out = out + bias
    if self.activation:
      out = self.activation(out)
    out = jnp.reshape(out, (*original_shape[0:-1], out.shape[-1]))
  
    return out


