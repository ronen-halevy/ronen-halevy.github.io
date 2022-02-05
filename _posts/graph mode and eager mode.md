
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

from https://github.com/zzh8829/yolov3-tf2 :



@tf.function
@tf.function is very cool. It's like an in-between version of eager and graph. You can step through the function by disabling tf.function and then gain performance when you enable it in production. Important note, you should not pass any non-tensor parameter to @tf.function, it will cause re-compilation on every call. I am not sure whats the best way other than using globals.

GradientTape
Extremely useful for debugging purpose, you can set breakpoints anywhere. You can compile all the keras fitting functionalities with gradient tape using the run_eagerly argument in model.compile. From my limited testing, all training methods including GradientTape, keras.fit, eager or not yeilds similar performance. But graph mode is still preferred since it's a tiny bit more efficient.

model(x) vs. model.predict(x)
When calling model(x) directly, we are executing the graph in eager mode. For model.predict, tf actually compiles the graph on the first run and then execute in graph mode. So if you are only running the model once, model(x) is faster since there is no compilation needed. Otherwise, model.predict or using exported SavedModel graph is much faster (by 2x). For non real-time usage, model.predict_on_batch is even faster as tested by @AnaRhisT94)


model(x) vs. model.predict(x)
