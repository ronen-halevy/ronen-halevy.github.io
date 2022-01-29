
https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/intro_to_graphs.ipynb#scrollTo=_Df6ynXcAaup

https://www.tensorflow.org/guide/migrate/tf1_vs_tf2 :

Performance: The function can be optimized (node pruning, kernel fusion, etc.)
Portability: The function can be exported/reimported (SavedModel 2.0 RFC), allowing you to reuse and share modular TensorFlow functions.

https://www.tensorflow.org/guide/migrate/tf1_vs_tf2 :

With the power to freely intersperse Python and TensorFlow code, you can take advantage of Python's expressiveness. However, portable TensorFlow executes in contexts without a Python interpreter, such as mobile, C++, and JavaScript. To help avoid rewriting your code when adding tf.function, use AutoGraph to convert a subset of Python constructs into their TensorFlow equivalents:

for/while -> tf.while_loop (break and continue are supported)
if -> tf.cond
for _ in dataset -> dataset.reduce

https://www.tensorflow.org/guide/migrate/tf1_vs_tf2 :

tf.function only supports singleton variable creations on the first call. To enforce this, when tf.function detects variable creation in the first call, it will attempt to trace again and raise an error if there is variable creation in the second trace.

...
The most straightfoward solution is ensuring that the variable creation and dataset creation are both outside of the tf.funciton call. For example:


