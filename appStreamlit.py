import streamlit as st
from funciones import *
import os
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import json

movies_soup = pd.read_parquet('./input/movies_with_credits.parquet')
movies_with_genre = pd.read_parquet('./input/movies_with_genre.parquet')
ratings_sample = pd.read_parquet('./input/ratings_sample.parquet')
ratings_sample = ratings_sample[ratings_sample['movieId'].isin(movies_soup['id'])]
ratings_sample = ratings_sample[ratings_sample['movieId'].isin(movies_with_genre['id'])]
tags_matrix = sparse.load_npz('./input/tags_matrix.npz')
similarity_matrix = cosine_similarity(tags_matrix,tags_matrix)
_, modelKNN = dump.load('Modelos/modelKNN')
_, modelSVD = dump.load('Modelos/modelSVD')

dict = {'anoguera':1,
        'aalfie':2,
        'larbues':3
        }

# Inicializar el estado de sesión si aún no existe
if 'current_option' not in st.session_state:
    st.session_state['current_option'] = None

# # Mostrar el mensaje de bienvenida
st.write("¡Hola! Elige una de las siguientes opciones:")

# Opciones
options = ['Usuario', 'Género', 'Película']
user_choice = st.selectbox("Elige una opción:", [""] + options)

# # Manejar la selección del usuario
# if user_choice:
#     st.session_state['current_option'] = user_choice.lower()
#     st.write(f"Has seleccionado: {user_choice}")

# # Solicitar entrada adicional basada en la opción
# if st.session_state['current_option'] == 'usuario':
#     user_id = st.text_input("Ingresa tu nombre de usuario para una recomendación personalizada.")
# elif st.session_state['current_option'] == 'genero':
#     genre = st.text_input("Ingresa un género para obtener una recomendación.")
# elif st.session_state['current_option'] == 'pelicula':
#     movie_name = st.text_input("Ingresa el nombre de una película para obtener una recomendación similar.")

# # Botón para enviar la solicitud
# if st.button("Enviar"):
#     st.write("Procesando tu solicitud...")

#     try:
#         if st.session_state['current_option'] == 'usuario':
#             user_id = dict.get(user_id, None)  # Obtener el ID de usuario del diccionario
#             if user_id is not None:
#                 titles = recommender_hybrid(ratings_sample, movies_soup, similarity_matrix, movies_with_genre, modelKNN, modelSVD, user_id, 5)
#             else:
#                 st.write("Nombre de usuario no encontrado.")
#         elif st.session_state['current_option'] == 'genero':
#             titles = recommender_genre(movies_with_genre, genre, 5)
#         elif st.session_state['current_option'] == 'pelicula':
#             titles = recommender(movies_soup, similarity_matrix, movie_name, 5)

#         # Mostrar las recomendaciones
#         if titles:
#             st.write("Te recomiendo las siguientes películas:")
#             for title in titles:
#                 st.write(f"- {title}")
#         else:
#             st.write("No se encontraron recomendaciones.")

#     except Exception as e:
#         st.write(f"Error: {e}")

#     # Reiniciar el estado
#     st.session_state['current_option'] = None


# Health check route
@st.cache()
def health_check():
    return True

# Health check route
if st.checkbox('Health Check'):
    health_status = health_check()
    if health_status:
        st.success('Health check passed!')
    else:
        st.error('Health check failed.')
