import tensorflow as tf
from keras.layers import core
import EinsumDenseFp8 import as customized

tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True) # Enable XLA.

# Please tweak use_variable, if set to False, you will see fp8 gemm, otherwise not. It's not expected.
class TestModel(tf.keras.Model):
    def build(self, ouptut_shape):
        self.einsumdense=customized.EinsumDenseFp8('abc,cde->abde',output_shape=(16,48,64), use_variable=True)
    def call (self, inputs):
        x =self.einsumdense(inputs)
        return x

model = TestModel()
bs = 96
# all dims are multiple of 16
x_data = tf.random.normal(shape=(bs, 16, 32))
y_data = tf.random.normal(shape=(bs, 16, 48, 64))

tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True) # Enable XLA.

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(),
    metrics=["accuracy"],
    # run_eagerly=True,
    jit_compile=True
)

history = model.fit(x_data, y_data, batch_size=32, epochs=20,
                    validation_split=0.2, verbose=1)
