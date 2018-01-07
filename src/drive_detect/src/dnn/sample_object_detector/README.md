# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

@page dwx_object_detector_sample Object Detector Sample


The Object Detector sample streams an H.264 video and runs DNN inference on each frame to
detect objects. You can choose between Caffe and
NVIDIA<sup>&reg;</sup> TensorRT<sup>&tm;</sup> models.

![Car detector on a single H.264 stream](sample_object_detector.png)

For related object-detector samples, see:
- @ref dwx_object_dwdetector. This sample is a simple, low resolution, single-class sample that
  uses GoogLeNet architecture to show how to integrate a deep neural network (DNN)
  into DriveWorks to perform object detection. This sample is trained on a small
  amount of object detection data.
- @ref dwx_object_tracker_drivenet_sample, for a more sophisticated, higher resolution, multi-class sample.

The interpretation of the output of a network depends on the network design. In this sample,
2 output layers are interpreted as coverage and bounding boxes.

#### Running the Sample

There are 2 possible ways to run this sample.

The command line for the sample for Caffe is:

    ./sample_object_detector --video=<video file.h264>
                             --caffe_prototxt=<deploy file.prototxt> --caffe_model=<weights file.caffemodel>


@note The sample ignores the batch size in the deploy file.

The command line for the sample for TensorRT is:

    ./sample_object_detector --video=<video file.h264> --tensorRT_model=<TensorRT model file>

#### Output

The sample creates a window, displays a video, and overlays bounding boxes for detected objects.
