 tf.io.gfile
 https://www.tensorflow.org/api_docs/python/tf/io/gfile/GFile
 
 The main roles of the tf.io.gfile module are:

To provide an API that is close to Python's file I/O objects, and
To provide an implementation based on TensorFlow's C++ FileSystem API.
The C++ FileSystem API supports multiple file system implementations, including local files, Google Cloud Storage (using a gs:// prefix, and HDFS (using an hdfs:// prefix). TensorFlow exports these as tf.io.gfile, so that you can use these implementations for saving and loading checkpoints, writing to TensorBoard logs, and accessing training data (among other uses). However, if all your files are local, you can use the regular Python file API without any problem.

 with Image.open(path) as image:
 
 vs equivalent:
 
   img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
