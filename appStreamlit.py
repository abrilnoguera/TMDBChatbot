import streamlit as st
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset
from surprise import dump
# Importa cualquier otra biblioteca que necesites

# Carga de datos y modelos
# movies_soup = pd.read_parquet('./input/movies_with_credits.parquet')
movies_with_genre = pd.read_parquet('./input/movies_with_genre.parquet')
ratings_sample = pd.read_parquet('./input/ratings_sample.parquet')
# ratings_sample = ratings_sample[ratings_sample['movieId'].isin(movies_soup['id'])]
ratings_sample = ratings_sample[ratings_sample['movieId'].isin(movies_with_genre['id'])]
# tags_matrix = sparse.load_npz('./input/tags_matrix.npz')
# similarity_matrix = cosine_similarity(tags_matrix, tags_matrix)
_, modelKNN = dump.load('Modelos/modelKNN')
_, modelSVD = dump.load('Modelos/modelSVD')

# Función para generar recomendaciones
def recommender_surprise(data, movies, model, user_id, n=10):
    # Cargar el conjunto de datos en el formato de Surprise
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()

    # Obtener la lista de todos los ítems (películas) del conjunto de entrenamiento
    items = trainset.all_items()
    item_map = trainset.to_raw_iid

    # Obtener la lista de ítems (películas) que el usuario ya ha calificado
    user_rated_items = set([j for (j, _) in trainset.ur[trainset.to_inner_uid(user_id)]])

    # Predecir calificaciones para todos los ítems que el usuario no ha calificado aún
    predictions = [model.predict(user_id, item_map(i)) for i in items if i not in user_rated_items]

    # Ordenar las predicciones basadas en la calificación estimada
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Asegura que 'n' no sea mayor que la cantidad de películas disponibles
    n = min(n, len(predictions))

    # Obtener las top 'n' recomendaciones de películas
    top_n_recommendations = predictions[:n]

    # Mapear los IDs de las películas a sus títulos
    top_movies = pd.DataFrame([(
        pred.iid,
        movies[movies['id'] == pred.iid]['title'].iloc[0] if not movies[movies['id'] == pred.iid].empty else 'Unknown',
        pred.est
    ) for pred in top_n_recommendations], columns=['itemID', 'title', 'estimatedRating'])

    return top_movies

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
                recomendaciones = recommender_surprise(ratings_sample, movies_with_genre, modelSVD, usuario_id, 5)
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