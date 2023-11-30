import streamlit as st
from funciones import recommender_hybrid, recommender_genre, recommender
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json

def main():
    # Carga de datos y modelos
    movies_soup = pd.read_parquet('./input/movies_with_credits.parquet')
    movies_with_genre = pd.read_parquet('./input/movies_with_genre.parquet')
    ratings_sample = pd.read_parquet('./input/ratings_sample.parquet')
    ratings_sample = ratings_sample[ratings_sample['movieId'].isin(movies_soup['id'])]
    ratings_sample = ratings_sample[ratings_sample['movieId'].isin(movies_with_genre['id'])]
    tags_matrix = sparse.load_npz('./input/tags_matrix.npz')
    similarity_matrix = cosine_similarity(tags_matrix, tags_matrix)
    _, modelKNN = dump.load('Modelos/modelKNN')
    _, modelSVD = dump.load('Modelos/modelSVD')

    user_dict = {'anoguera':1, 'aalfie':2, 'larbues':3}

    # Interfaz de usuario
    st.title("Chatbot de Recomendación de Películas")
    st.write("¡Hola! Elige una de las siguientes opciones:")
    options = ['Usuario', 'Género', 'Película']
    user_choice = st.selectbox("Elige una opción:", [""] + options)

    # Manejo de opciones
    user_input = ''
    if user_choice == 'Usuario':
        user_input = st.text_input("Ingresa tu nombre de usuario para una recomendación personalizada.")
    elif user_choice == 'Género':
        user_input = st.text_input("Ingresa un género para obtener una recomendación.")
    elif user_choice == 'Película':
        user_input = st.text_input("Ingresa el nombre de una película para obtener una recomendación similar.")

    # Botón de envío
    if st.button("Enviar") and user_input:
        try:
            if user_choice == 'Usuario':
                user_id = user_dict.get(user_input, None)
                titles = recommender_hybrid(ratings_sample, movies_soup, similarity_matrix, movies_with_genre, modelKNN, modelSVD, user_id, 5) if user_id else ["Nombre de usuario no encontrado."]
            elif user_choice == 'Género':
                titles = recommender_genre(movies_with_genre, user_input, 5)
            elif user_choice == 'Película':
                titles = recommender(movies_soup, similarity_matrix, user_input, 5)

            st.write("Te recomiendo las siguientes películas:")
            for title in titles:
                st.write(f"- {title}")
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
