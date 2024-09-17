import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.pipeline import Pipeline
from skimage import io
import streamlit as st

url = 'https://avatars.mds.yandex.net/i?id=2695191a6a6c85c2c8d65bf6ae492e37_l-5234389-images-thumbs&n=13'

image = io.imread(url)[:, :, 0]
plt.imshow(image, cmap='gray')

# загрузка файла
st.title('SVD')
uploaded_file = st.file_uploader("Загрузите изображение",
                                 type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # загрузка изображения
    image = io.imread(uploaded_file)[:, :, 0]
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    st.pyplot(fig)

    # plt.imshow(image, cmap='gray')
    # st.pyplot()

U, sing_values, V = np.linalg.svd(image)

sigma = np.zeros(shape=image.shape)
np.fill_diagonal(sigma, sing_values)

st.write(f'Размерность массивов: U = {U.shape}, sing_values = {sigma.shape}, V = {V.shape}')

# fig, ax = plt.subplots()
# ax.imshow(U@sigma@V, cmap='gray')
# st.pyplot(fig)

top_k = st.number_input('Введите кол-во сингулярных чисел', min_value=1, max_value=1920)

trunc_U = U[:, :top_k]
trunc_sigma = sigma[:top_k, :top_k]
trunc_V = V[:top_k, :]

st.write(f'Размерность массивов top_k сингулярных чисел: U = {trunc_U.shape}, sing_values = {trunc_sigma.shape}, V = {trunc_V.shape}')

st.title('Результат разложения матрицы по SVD')

fig, axes = plt.subplots(1, 2, figsize=(15, 10))

axes[0].imshow(io.imread(uploaded_file)[:, :, 0], cmap='gray')
axes[0].set_title('Исходное изображение')
axes[1].imshow(trunc_U@trunc_sigma@trunc_V, cmap='gray')
axes[1].set_title(f'{top_k} сингулярных чисел')

st.pyplot(fig)

st.write(f'Картинка минимально различима при top_k равным 5. Если выбирать меньшее число сингулярных чисел, то восприятие картинки практически невозможно')
st.write(f'Доля сингулярных чисел 5/1920')







