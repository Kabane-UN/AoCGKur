import csv
from time import sleep

import cv2
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

path = "images"
IMAGE_FILES = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
    data = []
    maxh, maxw = 531, 689
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(os.path.join(path, file))
        
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            continue
        height, width, _ = image.shape
        for face_landmarks in results.multi_face_landmarks:
            top_y = round(face_landmarks.landmark[27].y * height)
            left_x = round(face_landmarks.landmark[130].x * width)
            bottom_y = round(face_landmarks.landmark[23].y * height)
            right_x = round(face_landmarks.landmark[133].x * width)
            iris_x, iris_y = round(face_landmarks.landmark[468].x * width), round(
                face_landmarks.landmark[468].y * height)
            diffh = maxh-(bottom_y-top_y)
            diffw = maxw-(right_x-left_x)
            top_y -= round(diffh/2)
            bottom_y += diffh-round(diffh/2)
            left_x -= round(diffw/2)
            right_x += diffw-round(diffw/2)
            iris_x -= left_x
            iris_y -= top_y
            angel_x, angel_y = round(face_landmarks.landmark[471].x * width) - left_x, round(
                 face_landmarks.landmark[471].y * height) - top_y
            image = image[top_y:bottom_y, left_x:right_x]
            image = cv2.resize(image, (230, 177), interpolation=cv2.INTER_AREA)
            h, w, _ = image.shape
            iris_x /= w*3
            iris_y /= h*3
            angel_x /= w*3
            angel_y /= h*3
            #rad = ((iris_x - angel_x) ** 2 + (iris_y - angel_y) ** 2) ** .5
            data.append([file[:-4], iris_x, iris_y, angel_x, angel_y])
            # iris_x = round(iris_x*w)
            # iris_y = round(iris_y*h)
            # rad = round(rad*w)
            # cv2.circle(image, (iris_x, iris_y), rad, (255, 0, 0))
            # cv2.imshow("res", image)
            # cv2.waitKey(100)
        cv2.imwrite('dataset/images/' + file, image)
    with open(r'dataset/labels.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerows(data)
