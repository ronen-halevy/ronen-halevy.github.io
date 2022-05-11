https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564

https://www.tensorflow.org/tutorials/load_data/tfrecord

from # Better performance with tf.function: 
Reading data from files via TFRecordDataset, CsvDataset, etc. is the most effective way to consume data, as then TensorFlow itself can manage the asynchronous loading and prefetching of data, without having to involve Python. To learn more, see the tf.data: Build TensorFlow input pipelines guide.
