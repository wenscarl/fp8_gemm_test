import tensorflow as tf
from keras.layers import core
import EinsumDenseFp8 as customized

tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True) # Enable XLA.

B, C, D, E = 128, 64, 72, 48
# Please tweak use_variable, if set to False, you will see fp8 gemm, otherwise not. It's not expected.
class TestModel(tf.keras.Model):
    def build(self, ouptut_shape):
        self.einsumdense=customized.EinsumDenseFp8('abc,cde->abde', output_shape=(B, D, E))
        self.einsumdense2=customized.EinsumDenseFp8('abde,cde->abce', output_shape=(B, C, E),
                                                    is_last=True)
    def call (self, inputs):
        x = self.einsumdense(inputs)
        x = self.einsumdense2(x)
        return x

model = TestModel()
bs = 96
# all dims are multiple of 16
x_data = tf.random.normal(shape=(bs, B, C))
y_data = tf.random.normal(shape=(bs, B, C, E))

tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True) # Enable XLA.

model.compile(
    loss='mean_squared_error',
    optimizer=tf.keras.optimizers.RMSprop(),
    metrics=["accuracy"],
    # run_eagerly=True,
    jit_compile=True
)

history = model.fit(x_data, y_data, batch_size=32, epochs=20,
                    validation_split=0.0, verbose=1)
