import numpy as np
import tensorflow as tf
import cv2

import numpy as np
from numpy import linalg as LA
from datetime import datetime
import math

from pose_estimation import utils
from pose_estimation.data import BodyPart

import requests
import time


from pose_estimation.ml import Movenet
movenet = Movenet('movenet_thunder')

# Gọi hàm nhận dạng của movenet nhiều lần để cải thiện độ chính xác nhận dạng khung xương
def detect(input_tensor, inference_count=3):
  """Runs detection on an input image.
 
  Args:
    input_tensor: A [height, width, 3] Tensor of type tf.float32.
    inference_count: Số lần lặp.
 
  Returns:
    A Person entity detected by the MoveNet.SinglePose.
  """
  image_height, image_width, channel = input_tensor.shape

  # Detect pose using the full input image
  movenet.detect(input_tensor.numpy(), reset_crop_region=True)

  # Repeatedly using previous detection result to identify the region of
  # interest and only croping that region to improve detection accuracy
  for _ in range(inference_count - 1):
    person = movenet.detect(input_tensor.numpy(),
                            reset_crop_region=False)

  return person


def get_center_point(landmarks, left_bodypart, right_bodypart):
    """Calculates the center point of the two given landmarks."""

    left = tf.gather(landmarks, left_bodypart.value, axis=1)
    right = tf.gather(landmarks, right_bodypart.value, axis=1)
    center = left * 0.5 + right * 0.5
    return center


def get_pose_size(landmarks, torso_size_multiplier=2.5):
    """Calculates pose size.

    It is the maximum of two values:
      * Torso size multiplied by `torso_size_multiplier`
      * Maximum distance from pose center to any pose landmark
    """
    # Hips center
    hips_center = get_center_point(landmarks, BodyPart.LEFT_HIP,
                                   BodyPart.RIGHT_HIP)

    # Shoulders center
    shoulders_center = get_center_point(landmarks, BodyPart.LEFT_SHOULDER,
                                        BodyPart.RIGHT_SHOULDER)

    # Torso size as the minimum body size
    torso_size = tf.linalg.norm(shoulders_center - hips_center)

    # Pose center
    pose_center_new = get_center_point(landmarks, BodyPart.LEFT_HIP,
                                       BodyPart.RIGHT_HIP)
    pose_center_new = tf.expand_dims(pose_center_new, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to
    # perform substraction
    pose_center_new = tf.broadcast_to(pose_center_new,
                                      [tf.size(landmarks) // (17*2), 17, 2])

    # Dist to pose center
    d = tf.gather(landmarks - pose_center_new, 0, axis=0,
                  name="dist_to_pose_center")
    # Max dist to pose center
    max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))

    # Normalize scale
    pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)

    return pose_size


def feature_pose(landmarks):
    """Normalizes the landmarks translation by moving the pose center to (0,0) and
    scaling it to a constant pose size.
    """
    # Move landmarks so that the pose center becomes (0,0)
    pose_center = get_center_point(landmarks, BodyPart.LEFT_HIP,
                                   BodyPart.RIGHT_HIP)
    pose_center = tf.expand_dims(pose_center, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to perform
    # substraction
    pose_center = tf.broadcast_to(pose_center,
                                  [tf.size(landmarks) // (17*2), 17, 2])
    landmarks = landmarks - pose_center

    # Scale the landmarks to a constant pose size
    pose_size = get_pose_size(landmarks)
    landmarks /= pose_size

    return landmarks


def normalize_drop_score(landmarks_and_scores):
    """Converts the input landmarks into a pose embedding."""
    # Reshape the flat input into a matrix with shape=(17, 3)
    reshaped_inputs = tf.reshape(landmarks_and_scores, [1, 17, 3])

    # Normalize landmarks 2D
    norm_landmarks = feature_pose(reshaped_inputs[:, :, :2])

    return norm_landmarks


def angle_between_two_vector(a, b):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    inner = np.inner(a, b)
    norms = LA.norm(a) * LA.norm(b)
    cos = inner / norms
    return np.arccos(np.clip(cos, -1.0, 1.0))


def flatten(x):
    return x.numpy().flatten()


def angle_between_three_point(landmarks, bodypart1, bodypart2, bodypart3, isBodyPart=True):
    if isBodyPart:
        bodypart1 = tf.gather(landmarks, bodypart1.value, axis=1)
        bodypart2 = tf.gather(landmarks, bodypart2.value, axis=1)
        bodypart3 = tf.gather(landmarks, bodypart3.value, axis=1)

    v21 = bodypart1 - bodypart2
    v23 = bodypart3 - bodypart2

    return angle_between_two_vector(flatten(v21), flatten(v23))


def feature_angle(landmarks):
    center_ear = get_center_point(
        landmarks, BodyPart.LEFT_EAR, BodyPart.RIGHT_EAR)
    center_shoulder = get_center_point(
        landmarks, BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER)
    center_hip = get_center_point(
        landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)

    angle_ear_shoulder = angle_between_three_point(
        landmarks, center_ear, center_shoulder, center_hip, False)

    angle_left_torso_thighs = angle_between_three_point(
        landmarks, BodyPart.LEFT_SHOULDER, BodyPart.LEFT_HIP, BodyPart.LEFT_KNEE)

    angle_right_torso_thighs = angle_between_three_point(
        landmarks, BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE)

    angle_left_thighs_tibia = angle_between_three_point(
        landmarks, BodyPart.LEFT_HIP, BodyPart.LEFT_KNEE, BodyPart.LEFT_ANKLE)

    angle_right_thighs_tibia = angle_between_three_point(
        landmarks, BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE, BodyPart.RIGHT_ANKLE)

    backbone_vector = flatten(center_shoulder - center_hip)
    horizontal_backbone_angle = angle_between_two_vector(backbone_vector, (1,0))

    return [horizontal_backbone_angle, angle_ear_shoulder, angle_left_torso_thighs, angle_left_thighs_tibia, angle_right_torso_thighs, angle_right_thighs_tibia]


def get_distance(landmarks, left, right):
    left = tf.gather(landmarks, left.value, axis=1)
    right = tf.gather(landmarks, right.value, axis=1)

    vector2 = (right - left).numpy()[0] ** 2

    return math.sqrt(vector2[0] + vector2[1])

def feature_extract(landmarks):
    landmarks = normalize_drop_score(landmarks)
    feature_vector = feature_angle(landmarks)
    feature_vector.append(get_distance(landmarks, BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER))
    pose = feature_pose(landmarks).numpy().tolist()

    unwanted = [BodyPart.LEFT_ELBOW.value, BodyPart.RIGHT_ELBOW.value, BodyPart.LEFT_WRIST.value, BodyPart.RIGHT_WRIST.value]
    for index in sorted(unwanted, reverse = True): 
        del pose[0][index]

    wanted_point = []

    for item in pose[0]:
        for i in item:
            wanted_point.append(i)

    feature_vector.extend(wanted_point)

    return feature_vector

def evaluate_model(interpreter, X):
  """Evaluates the given TFLite model and return its accuracy."""
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]

  # Run predictions on all given poses.
  interpreter.set_tensor(input_index, X)

  # Run inference.
  interpreter.invoke()

  # Post-processing: remove batch dimension and find the class with highest
  # probability.
  output = interpreter.tensor(output_index)

  predicted_label = np.argmax(output()[0])

  notes = ""
  for i in range(len(output()[0])):
    notes += f"{class_names[i]}: {round(output()[0][i]*100/output()[0].sum(), 5)} %\n"

  return predicted_label

# TEST
class_names = []
with open('pose_labels.txt','r') as f:
  for line in f.readlines():
    class_names.append(line.strip())

print(class_names)

y_pred = []

# Evaluate the accuracy of the converted TFLite model
classifier_interpreter = tf.lite.Interpreter(
    model_path='pose_classifier.tflite')
classifier_interpreter.allocate_tensors()

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, image = cap.read()

    if ret:
      # Extract pose from image
        tensor = tf.convert_to_tensor(image)
        person = detect(tensor)

        landmarks = []
        for keypoint in person.keypoints:
          landmarks.extend(
              [keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score])

        feature = feature_extract(tf.constant([landmarks]))
        feature = [list(map(np.float32, feature))]



        y_pred = evaluate_model(classifier_interpreter, feature)

        # Draw the detection result on top of the image.
        image_np = utils.visualize(image, [person])

        # Plot the image with detection results.
        # offsetbox = TextArea(notes)

        # ab = AnnotationBbox(offsetbox, (0.5, 0.7),
        #                     xybox=(0.95, 0.5),
        #                     # xycoords='data',
        #                     boxcoords=("axes fraction", "data"),
        #                     # box_alignment=(0., 0.5),
        #                     arrowprops=dict(arrowstyle="->"))
        # ax.add_artist(ab)

        cv2.putText(image_np, class_names[y_pred], (int(
            image.shape[0]*0.35), int(image.shape[1]*0.5)), None, 1.5, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.imwrite('./temp.jpg', image)

        files = {
           "image": open('./temp.jpg', "rb")
        }

        payload = {
            "posture": class_names[y_pred],
            "date": datetime.now().__str__()
        }
        
        response = requests.post('https://pbl5server.onrender.com/api/img/pose', files=files, data=payload)
        # response = requests.post('http://localhost:8080/api/img/pose', files=files, data=payload)
        time.sleep(0.5)

        # Gõ q để tắt cam
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()