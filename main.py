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

st.set_page_config(
    page_title="Рисуем нейронной сетью",
    page_icon="🧊",
    initial_sidebar_state="expanded"
)

page_style = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
text-align: center
}}
"""

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

def load_tensor_img(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

@st.cache
def load_model():
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    return hub_model
@st.cache
def plot_final(original_img, style_img):
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    final_img = tensor_to_image(stylized_image)
    return final_img

if __name__ == '__main__':

    st.markdown(page_style, unsafe_allow_html=True)

    instructions = """
        Нейронный перенос стиля — это метод, использующий два изображения — 
        изображения контента и эталонного изображения стиля 
        (например, произведения искусства известного художника) — 
        и их смешивания вместе, чтобы выходное изображение выглядело 
        как изображение контента, но «нарисовано» в стиле эталонного изображения стиля.

        Это реализуется путем обработки выходного изображения для 
        соответствия статистике содержимого изображения содержимого и 
        статистике стиля эталонного изображения стиля. 
        Эта статистика извлекается из изображений с помощью нейронной сверточной сети.
        """
    st.write(instructions)

    hub_model = load_model()

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
        st.subheader('Результат')
        stylized_image = hub_model(tf.constant(origin_tensor), tf.constant(style_tensor))[0]
        final_img = tensor_to_image(stylized_image)

        st.image(final_img)

    elif not ready_style and  not ready_origin:
        with original:
            img = '325px-Kramskoy_Portrait_of_a_Woman.jpg'
            img_path = 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/Kramskoy_Portrait_of_a_Woman.jpg/325px-Kramskoy_Portrait_of_a_Woman.jpg'

            content_path = tf.keras.utils.get_file(img, img_path)
            st.image(content_path)

            content_image = load_img(content_path)
            content_image = preprocess_img(content_image)

            # imshow(content_image, 'Оригинал')

        with style:
            style = 'Vassily_Kandinsky%2C_1913_-_Composition_7.jpg'
            style_path = 'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg'

            style_path = tf.keras.utils.get_file(style, style_path)
            st.image(style_path, )

            style_image = load_img(style_path)
            style_image = preprocess_img(style_image)

            # imshow(content_image, 'Оригинал')

        st.subheader('Результат')


        stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
        final_img = tensor_to_image(stylized_image)

        st.image(final_img)
