import os
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from keras.applications import VGG19
from keras.layers import Dense, Flatten, Input
from keras.optimizers import Adam
from keras.models import Model
import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

images_path = "dataset/images"
coords_path = "dataset/labels.csv"
images_names = os.listdir(images_path)
csv_data = {}
with open(coords_path, "r") as file:
    reader = csv.reader(file)
    for row in reader:
        csv_data[row[0]] = [float(row[1]), float(row[2]), float(row[3])]
images = []
coords = []
for image_name in images_names:
    img = cv2.imread(os.path.join(images_path, image_name))
    coord = csv_data[image_name[:-4]]
    img = img_to_array(img)
    images.append(img)
    coords.append(coord)
images = np.array(images, dtype="float32") / 255.0
coords = np.array(coords, dtype="float32")
split = train_test_split(images, coords, test_size=0.2, random_state=123)
(train_images, test_images) = split[:2]
(train_coords, test_coords) = split[2:]

vgg = VGG19(weights="imagenet", include_top=False, input_tensor=Input(shape=(33, 63, 3)))
vgg.trainable = False
flatten = Flatten()(vgg.output)
pr_coord = Dense(64, activation="relu")(flatten)
pr_coord = Dense(32, activation="relu")(pr_coord)
pr_coord = Dense(3, activation="sigmoid")(pr_coord)

model = Model(inputs=vgg.inputs, outputs=pr_coord)
model.compile(loss="mse", optimizer=Adam())
history = model.fit(
    train_images, train_coords,
    validation_data=(test_images, test_coords),
    batch_size=32,
    epochs=25,
    verbose=1)
