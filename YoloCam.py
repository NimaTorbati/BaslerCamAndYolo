from arrayqueues.shared_arrays import ArrayQueue, ArrayView
from multiprocessing import Process
import numpy as np
import pypylon.pylon as py
from time import sleep
from time import time
import torch
from ultralytics import YOLO
from PIL import Image


#save function
save_path = 'F:/Camera/cuda/'
def save1(im):
    ind = 0
    print("new_defect")
    print("save1")
    ind = ind + 1
    if (ind == 100):
        ind = 0
    im = Image.fromarray(im)  # RGB PIL image
    im.save(save_path + 'results' + str(ind) + '.jpg') 


# set GPU if available
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    dev = 'cuda'
else:
    dev = 'cpu'
dev = 'cuda'
print(dev)

#load yolo cofs
ov_model = YOLO('F:/Camera/first_try/best.pt', task = 'detect')
ov_model.to(device = dev)

#run model once
frame = np.zeros((1000, 2464, 3))
print(np.shape(frame))
results = ov_model.predict(source=frame,conf=0.35,iou=0.75,show_labels = False, show_conf = False, device = dev)

#start camera
cam = py.InstantCamera(py.TlFactory.GetInstance().CreateFirstDevice())
cam.Open()

#set camera features
maxWidth = cam.WidthMax.GetValue()#maximum width
#cam.Width.SetValue(2456)# frame width is set equal to 2456
#cam.Height.SetValue(500)# frame height is equal to 500
cam.OffsetX.SetValue(0)# This parameter defines the starting pixel of ROI in x-axis
cam.OffsetY.SetValue(0)# this parameter defines the starting pixel of ROI in y-axis

#print(cam.ExposureTime.Value)

cam.ExposureMode.SetValue("Timed")
cam.ExposureAuto.SetValue("Off")
#cam.ExposureTimeMode.SetValue("UltraShort")# not available
cam.ExposureTime.SetValue(10000.0)# the exposure time to capture the frame, the lowe0r the value the higher the frame rate,
# please note that a very low exposure time results in a dark frame

maxWidth = cam.WidthMax.GetValue()#maximum width
maxHeight = cam.HeightMax.GetValue()
cam.PixelFormat.SetValue("RGB8")

cam.Width.SetValue(maxWidth)
cam.Height.SetValue(2056)#(2056)

cam.AcquisitionFrameRateEnable.SetValue(True)
cam.AcquisitionFrameRate.SetValue(100.0)
cam.LightSourcePreset.SetValue("Daylight5000K")
ind = 1

#start grabbing frames
cam.StartGrabbing()
while True:
    with cam.RetrieveResult(500) as res:
        if res.GrabSucceeded():
            frame = res.GetArray()

            #run yolo
            results = ov_model.predict(source=frame,conf=0.35,iou=0.75, verbose = False,imgsz = 640)

            #if object detected save frame with objects
            if  (len(results[0].boxes.cls)) > 0:
                im_array = results[0].plot() #BGR numpy array of predictions
                print(np.shape(im_array))
                save1(im_array)
        else:
            print("failed")
            cam.Close()
            cam = py.InstantCamera(py.TlFactory.GetInstance().CreateFirstDevice())
            sleep(0.01)                
            cam.Open()
            cam.StartGrabbing()
            sleep(0.01)                
            ind = 0
cam.StopGrabbing()
cam.Close()



