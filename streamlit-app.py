import streamlit as st
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset
from surprise import dump
from funciones import *
# Importa cualquier otra biblioteca que necesites

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

dict = {'anoguera':1,
        'aalfie':2,
        'larbues':3
        }

# Interfaz de usuario con Streamlit
def main():
    st.title("Recomendador de Películas")

    usuario = st.text_input("Introduce tu ID de usuario", "")

    if st.button("Recomendar Películas"):
        if usuario:
            usuario_id = dict.get(usuario)
            if usuario_id is not None:
                recomendaciones = recommender_hybrid(ratings_sample, movies_soup, similarity_matrix, movies_with_genre, modelKNN, modelSVD, usuario_id, n=5)
                if not recomendaciones.empty:
                    st.write("Recomendaciones:")
                    for pelicula in recomendaciones['title']:
                        st.write(pelicula)
                else:
                    st.write("No se encontraron recomendaciones.")
            else:
                st.write("Usuario no encontrado. Por favor, introduce un nombre de usuario válido.")
        else:
            st.write("Por favor, introduce un nombre de usuario.")

if __name__ == "__main__":
    main()