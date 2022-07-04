import tensorflow as tf

graph = tf.Graph()
with graph.as_default():
  x = tf.compat.v1.placeholder(tf.float32, shape=[None, 200], name="input")
  W = tf.Variable(tf.random.truncated_normal([200, 100], stddev=0.1))
  b = tf.Variable(tf.ones([100], tf.float32))
  y = tf.matmul(x, W)
  out = tf.add(y, b, name = "output")
  output_names = [out.op.name]

import tempfile
import os 
from tensorflow.python.tools.freeze_graph import freeze_graph

model_dir = tempfile.mkdtemp()
graph_def_file = os.path.join(model_dir, 'tf_graph.pb')
checkpoint_file = os.path.join(model_dir, 'tf_model.ckpt')
frozen_graph_file = os.path.join(model_dir, 'tf_frozen.pb')

with tf.compat.v1.Session(graph=graph) as sess:
  # initialize variables
  sess.run(tf.compat.v1.global_variables_initializer())
  # save graph definition somewhere
  tf.io.write_graph(sess.graph, model_dir, graph_def_file, as_text=False)
  # save the weights
  saver = tf.compat.v1.train.Saver()
  saver.save(sess, checkpoint_file)

  # take the graph definition and weights 
  # and freeze into a single .pb frozen graph file
  freeze_graph(input_graph=graph_def_file,
               input_saver="",
               input_binary=True,
               input_checkpoint=checkpoint_file,
               output_node_names=",".join(output_names),
               restore_op_name="save/restore_all",
               filename_tensor_name="save/Const:0",
               output_graph=frozen_graph_file,
               clear_devices=True,
               initializer_nodes="")
  
print("TensorFlow frozen graph saved at {}".format(frozen_graph_file))
