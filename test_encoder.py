import tensorflow as tf
from official.nlp.modeling.layers.transformer_encoder_block import TransformerEncoderBlock

tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True) # Enable XLA.
num_attention_heads = 16
sequence_length = 64
width = 64

def test_return_attention_scores():
   tf.config.experimental.enable_tensor_float_32_execution(False)
   test_layer = TransformerEncoderBlock(
       num_attention_heads=num_attention_heads,
       inner_dim=2048,
       inner_activation='relu',
       return_attention_scores=False)
   return test_layer
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True) # Enable XLA.

def gen_model():
  return tf.keras.models.Sequential([test_return_attention_scores()])

model = gen_model()
def compile_model(model):
  opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
  model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
  return model


model = compile_model(model)

def train_model(model, x_train, y_train, epochs=25):
  model.fit(x_train, y_train, batch_size=16, epochs=epochs)

bs = 96
data_tensor = tf.random.normal(shape=(bs, sequence_length, width))
y_tensor = tf.random.normal(shape=(bs, sequence_length, width))
train_model(model, data_tensor, y_tensor)  
