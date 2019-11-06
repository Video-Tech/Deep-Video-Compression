from object_detection.yolo_opencv import detect_objects

image   = 'object_detection/dog.jpg'

objects = detect_objects(image)
