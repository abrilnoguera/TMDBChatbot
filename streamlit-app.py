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
# movies_with_genre = pd.read_parquet('./input/movies_with_genre.parquet')
ratings_sample = pd.read_parquet('./input/ratings_sample.parquet')
ratings_sample = ratings_sample[ratings_sample['movieId'].isin(movies_soup['id'])]
# ratings_sample = ratings_sample[ratings_sample['movieId'].isin(movies_with_genre['id'])]
tags_matrix = sparse.load_npz('./input/tags_matrix.npz')
similarity_matrix = cosine_similarity(tags_matrix, tags_matrix)
# _, modelKNN = dump.load('Modelos/modelKNN')
# _, modelSVD = dump.load('Modelos/modelSVD')

def recommender(df, similarity_matrix, movie, n=30):
    # Encuentra el índice de la película proporcionada en el DataFrame.
    idx = df.loc[df['title'] == movie, :].index[0]

    # Obtén los puntajes de similitud con otras películas.
    similarity_score = list(enumerate(similarity_matrix[idx]))

    # Ordena los puntajes de similitud en orden descendente.
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    # Asegura que 'n' no sea mayor que la cantidad de películas disponibles
    n = min(n, len(similarity_score))

    # Selecciona los puntajes de las 'n' películas más similares, excluyendo la primera que es la propia película introducida.
    similarity_score = similarity_score[1:n + 1]

    # Obtiene los índices de las películas más similares y sus puntajes de similitud.
    movie_indices = [i[0] for i in similarity_score]
    scores = [i[1] for i in similarity_score]

    # Crea un DataFrame con títulos de películas y sus puntajes de similitud.
    pd.set_option('display.float_format', '{:.2f}'.format)
    recommendation = pd.DataFrame({
        'title': df['title'].iloc[movie_indices],
        # 'similarity_score': scores
    })

    return recommendation

def recommender_movies(ratings, df, similarity_matrix, user_id, n=30):

    movies = ratings[ratings['userId'] == user_id].sort_values('rating', ascending=False)
    a = min(len(movies), 3)
    movies = movies['movieId'].head(a)
    if a == 0:
        return pd.DataFrame()
    recommendation = pd.DataFrame()
    for movie in movies:
        title = df[df['id']==movie]['title'].values[0]
        # Suprime solo los FutureWarnings específicos de Pandas
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            recommendation = pd.concat([recommendation, recommender(df, similarity_matrix, title, n*2)])
        recommendation.drop_duplicates(subset='title', keep='first', inplace=True)

    # Asegura que 'n' no sea mayor que la cantidad de películas disponibles
    n = min(n, len(recommendation))

    recommendation = recommendation.head(n)
    return recommendation

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
                recomendaciones = recommender_movies(ratings_sample, movies_soup, similarity_matrix, usuario_id, n=5)
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