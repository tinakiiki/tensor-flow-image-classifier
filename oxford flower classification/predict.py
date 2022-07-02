import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import logging
from PIL import Image
from tensorflow.keras.models import load_model

# silence info and warning logs
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

def read_flower_names():
    # read category classes
    with open('label_map.json', 'r') as f:
        class_names = json.load(f)
    return class_names

## image prediction using saved model
def predict(image_path,saved_model, top_k):
    # process image
    im = Image.open(image_path)
    im_array = np.asarray(im)
    image = tf.cast(im_array, tf.float32)
    image /= 255
    image_size = 224
    image = tf.image.resize(image, [image_size, image_size], preserve_aspect_ratio=False)
    image = np.expand_dims(image, axis = 0)
    # read in saved model
    ## load the saved classification model 
    model = load_model(saved_model, custom_objects={'KerasLayer':hub.KerasLayer}, compile=False)
    probabilities = model.predict(image)
    top_k = tf.math.top_k(probabilities[0].tolist(),k=top_k)
    proba = top_k.values.numpy().tolist()
    classes = top_k.indices.numpy().tolist()
    flower_names = [class_names[f'{i+1}'] for i in classes]
    print(proba, classes, flower_names)

def collect_input():
    # collect input variables from terminal
    parser = argparse.ArgumentParser(description='flower image classifier')
    parser.add_argument('image_path')
    parser.add_argument('saved_model')
    parser.add_argument('--top_k', dest="top_k", type=int)
    args = parser.parse_args()
    return args.image_path , args.saved_model, args.top_k

if __name__ == "__main__":
    class_names = read_flower_names()
    image_path, saved_model, top_k = collect_input()
    predict(image_path, saved_model, top_k)