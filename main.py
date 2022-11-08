import streamlit as st
import tensorflow as tf
import numpy as np
import PIL.Image
import time
import functools
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False


def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)


def load_img(path_to_img):
    img = PIL.Image.open(path_to_img)
    img = np.array(img)
    img = img / 255.
    return img


def preprocess_img(img):
    img = tf.convert_to_tensor(img, dtype=tf.float32)

    max_dim = 640
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    print(img)
    return img


def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)


if __name__ == '__main__':
    st.header('Перенос стиля изображения')

    original, style = st.columns(2)

    ready_origin, ready_style = False, False

    with original:
        origin_img = st.file_uploader('Загрузите изображение')
        if origin_img:
            origin_img = load_img(origin_img)
            plt.subplot(1, 2, 1)
            st.image(origin_img, 'Оригинал')
            origin_tensor = preprocess_img(origin_img)
            ready_origin = True

    with style:
        style_img = st.file_uploader('Загрузите стиль')
        if style_img:
            style_img = load_img(style_img)
            plt.subplot(1, 2, 2)
            st.image(style_img, 'Стиль')
            style_tensor = preprocess_img(style_img)
            ready_style = True

    if ready_origin and ready_style:
        hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        stylized_image = hub_model(tf.constant(origin_tensor), tf.constant(style_tensor))[0]
        final_img = tensor_to_image(stylized_image)

        st.image(final_img)
