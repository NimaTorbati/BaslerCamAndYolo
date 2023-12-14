# BaslerCamAndYolo
Run Basler camera and Yolov8 Deep neural network on different platforms, Intel CPU, Intel UHD GPU, Nvidia GPU, and jetson nano.
## Specs1
This notebook includes different acquisition specs of the Camera including frame rate, brightness adjustment and ...
## YoloSpeedTest
This notebook compares the performance of yolov8n on:
1. Intel Core i7 CPU  i7-8550U 1.80GHz
2. Intel UHD 620 graphic GPU
3. Nvidia GPU Geforce MX150 4G

**Note**: you should export the model to Openvino to get the best results for Intel processors.

Based on my results

  Rank1: Yolov8n model with Nvidia GPU =  25.65 FPS

  Rank2: Openvino model with Intel UHD 620 GPU =  11.98 FPS

  Rank3: Openvino model with CPU =  7.08 FPS
  
  Rank4: Yolov8n model with CPU =  4.55 FPS

  Rank6: Yolov8n model with jetson nano GPU = 4.3 FPS
  
  Rank5: Openvino model with Nvidia GPU =  2.35 FPS


## YoloCam
  This Python script runs the Basler camera and Yolo model together and saves the frame if an object is detected

## jetson nano
You can use the YoloCam script on Jetson Nano. 

It is important to follow the steps in: https://i7y.org/en/yolov8-on-jetson-nano/.

You can only run this script on python3.8 on jetson nano and only in virtual environment using venv.

I could reach 4.3 FPS using Jetson nano CUDA.

