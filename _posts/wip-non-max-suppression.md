When running an object detection algorithm, it frequently assigns multiple detection bounding boxes on a single object as as illustrated in the image below. Non Max Suppression algorithm cleans the multiple bounding boxes, leaving a single detection box per an object.

The algorithm selects the bounding box with max probability and suppresses all others.

Here's the algorithm:

- Repeat steps 1-2 untill there are no more unselected rectangles:
1. Select the rectangle with the largest probability
2. Suppress all rectangles with large IOU with the selected rectangle


Each bounding box is a assigned with a classification probability, as annotated in the illustration figure above.
0. Discard all detections with probability below a selected threshold.
1. Select the rectangle with the largest probability
2. Disacard all rectangles with large IOU with the selected rectangle
3. From the remaining boxes, select the rectangle with larges probability
4. Suppress all rectangles with large IOU with the selected rectangle
Repeat untill there are no more unselected rectangles.





