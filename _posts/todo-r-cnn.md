https://www.mihaileric.com/posts/object-detection-with-rcnn/
from there:
Final Thoughts
R-CNN fundamentally changed the landscape of object detection at the time of its conception, and it has radically influenced the design of modern-day detection algorithms.

In spite of that, one of the major shortcomings of R-CNN was the speed with which it worked. For example, as the original paper noted, simply computing region proposals and features would take 13 seconds per image on a GPU. This is clearly too slow for a real-time object detection algorithm.

Follow-up work sought to reduce the runtime of the model, especially time spent on computing region proposals. We will discuss these improved algorithms including Fast R-CNN, Faster R-CNN, and YOLO in subsequent blog posts.

In the meantime, if you are interested in further checking out some of R-CNN’s implementation details, you can see the original repo.
https://medium.com/@selfouly/r-cnn-3a9beddfd55a
https://learnopencv.com/selective-search-for-object-detection-cpp-python/

https://towardsdatascience.com/faster-rcnn-object-detection-f865e5ed7fc4#:~:text=Faster%20RCNN%20is%20an%20object,SSD%20(%20Single%20Shot%20Detector).

https://medium.com/@selfouly/r-cnn-3a9beddfd55a


https://www.mihaileric.com/posts/object-detection-with-rcnn/
