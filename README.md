# TensorRT based Real Time Object Classification using EfficientNet on Jetson Nano.

This repository contains Jupyter notebooks that contain step by step instructions for converting a pre-trained CNN model to ONNX and to TensorRT engine and do real time inferencing on camera input.

The purpose is for beginners to quickly understand the flow of converting and deploying a model using TensorRT on Nvidia Jetson Xavier series and for me to gain hands-on experience with TensorRT.

The Jupyter notebooks are split into self-contained notebooks, each focussing on a specific task. 
The notebooks do the following,

1. Downloading a pretrained CNN PyTorch model. By default EfficientNet-B2 is used. But the template can be reused for any model.
2. Do a test inference without TensorRT on the model.
3. Convert the model to ONNX format.
4. Convert the ONNX model to TensorRT engine and save to disk.
5. Do a test inference on the TensorRT engine.
6. Perform real time inferencing using TensorRT on camera input and display the class of object on the screen.

It also has a [Python script](https://github.com/arvcode/TensorRT_classifier_efficientNet/blob/main/TensorRT_efficientnet_infer.py) that can readily be used for Real Time Object Classification on Jetson Nano[4GB] board.

For executing the script, please git clone this repository and execute

```
python3.6 TensorRT_efficientnet_infer.py
```

Please note that the script by default uses camera 1.(ie) My board has 2 cameras (One MIPI-CSI2 and USB).

 Hence in the [Python script](https://github.com/arvcode/TensorRT_classifier_efficientNet/blob/main/TensorRT_efficientnet_infer.py), please change to 

```
print("Start Classifying...")
capture=cv.VideoCapture(1) #USB camera number 1.  << Change to 0 if there is only one camera.
print("Camera started...")
```



 
For setting up the environment to run Jupyter notebooks, please refer to the [jetson_nano_setup_instructions.md](https://github.com/arvcode/TensorRT_classifier_efficientNet/blob/main/jetson_nano_setup_instructions.md) in this repository.

Please also see the [References.md](https://github.com/arvcode/TensorRT_classifier_efficientNet/blob/main/References.md) file for the list of references.

Please also visit the blog post [Medium_blog](https://s-arvindh.medium.com/tensorrt-based-real-time-object-classification-using-efficientnet-on-jetson-nano-dcb6cc7a95f5).
Thank you!
