import cv2 as cv
import tensorflow as tf
import tensorflow.lite as tf_lite
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.collections import LineCollection

from pose_estimation import utils
from pose_estimation.data import BodyPart
from pose_estimation.ml import Movenet
movenet = Movenet('movenet_thunder')

def detect(input_tensor, inference_count=3):
    
    movenet.detect(input_tensor.numpy(), reset_crop_region=True)
    
    for _ in range(inference_count - 1):
        person = movenet.detect(input_tensor.numpy(), reset_crop_region=False)

    return person

def draw_prediction_on_image(
    image, person, crop_region=None):
  
    image_np = utils.visualize(image, [person])

    return image_np


cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # cv.imshow('frame', frame) 
    tensor = tf.convert_to_tensor(frame)
    # print(frame)
    person = detect(tensor)

    image = draw_prediction_on_image(image=frame, person=person)

    cv.imshow("", image)

    if cv.waitKey(1) & 0xFF == ord('q'): 
        break 

cap.release()

cv.destroyAllWindows()