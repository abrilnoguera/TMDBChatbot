import re
import ast
import pandas as pd
import numpy as np
from surprise import Reader, Dataset,accuracy
import time
import math
from tabulate import tabulate
import difflib
import random
import warnings
import dump


# Preparacion de la Base de Datos:
def extract_name(s):
    if isinstance(s, str):
        match = re.search(r"'name':\s*'([^']*)'", s)
        if match:
            return match.group(1)
    return None

def get_names(column):
    # Aplica ast.literal_eval para interpretar las cadenas como listas de diccionarios
    column = column.apply(lambda x: ast.literal_eval(x) if not pd.isna(x) else x)

    # Extrae los nombres de la lista de diccionarios
    column = column.apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

    return column

def get_director(column):
    # Asegúrate de que los datos están en formato de lista
    column = column.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Ahora itera sobre la lista y encuentra el director, asegurándote de que la entrada sea iterable
    director = column.apply(lambda crew: next((i['name'] for i in crew if i['department'] == 'Directing'), np.nan) if isinstance(crew, list) else np.nan)

    return director


# Content Based Recomendator para Soup:
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

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
    recomendation = pd.DataFrame({
        'title': df['title'].iloc[movie_indices],
        'similarity_score': scores
    })

    return recomendation

def recommender_movies(ratings, df, similarity_matrix, user_id, n=30):

    movies = ratings[ratings['userId'] == user_id].sort_values('rating', ascending=False)
    a = min(len(movies), 3)
    movies = movies['movieId'].head(a)
    if a == 0:
        return pd.DataFrame()
    recomendation = pd.DataFrame()
    for movie in movies:
        title = df[df['id']==movie]['title'].values[0]
        # Suprime solo los FutureWarnings específicos de Pandas
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            recomendation = pd.concat([recomendation, recommender(df, similarity_matrix, title, n*2)])
        recomendation.drop_duplicates(subset='title', keep='first', inplace=True)

    # Asegura que 'n' no sea mayor que la cantidad de películas disponibles
    n = min(n, len(recomendation))

    recomendation = recomendation.head(n)
    return recomendation


# Content Based Recomendator segun Popularidad:
def recommender_genre(df, genre, n=30):
    # Filtra películas por género
    mask = df.genres.apply(lambda x: genre in x)
    filtered_movie = df[mask]

    # Ordena las películas filtradas por popularidad
    filtered_movie = filtered_movie.sort_values(by='popularity', ascending=False)

    # Asegura que 'n' no sea mayor que la cantidad de películas disponibles
    n = min(n, len(filtered_movie))

    # Selecciona los 'n' mejores movieIds
    top_movie_ids = filtered_movie['movieId'].head(n).values.tolist()

    # Crea un DataFrame con títulos e IDs
    recomendation = pd.DataFrame({
        'title': [df[df['movieId'] == movie_id]['title'].values[0] for movie_id in top_movie_ids],
        'id': top_movie_ids
    })

    return recomendation

def InfoUser(movies_with_genre):
    unique_genre = movies_with_genre['genres'].explode().unique()

    # Make a dict assigning an index to a genre
    genre_dict = {k: v for v, k in enumerate(unique_genre)}

    idx_to_genre = {v: k for k, v in genre_dict.items()}
    user_ids = movies_with_genre['userId'].unique()

    user_df = pd.DataFrame(columns=['userId', 'user_vector', 'avg_rating', 'num_movies_rated'])

    for user_id in user_ids:
        user_rating_df = movies_with_genre[(movies_with_genre['userId'] == user_id)]

        user_vector = np.zeros(len(genre_dict))
        count_vector = np.zeros(len(genre_dict))

        user_avg_rating = 0
        movies_rated_count = 0

        for _, row in user_rating_df.iterrows():
            user_avg_rating += row.rating
            movies_rated_count += 1
            genres = row.genres

            # Check if genres is a float (non-iterable)
            if not isinstance(genres, float):
                user_movie_vector = np.zeros(len(genre_dict))

                for g in genres:
                    user_movie_vector[genre_dict[g]] = 1
                    count_vector[genre_dict[g]] += 1

                user_vector += user_movie_vector * row.rating

        count_vector = np.where(count_vector == 0, 1, count_vector)
        user_vector = np.divide(user_vector, count_vector)
        user_avg_rating /= movies_rated_count
        row_df = pd.DataFrame([[user_id, user_vector, user_avg_rating, movies_rated_count]],
                            columns=['userId', 'user_vector', 'avg_rating', 'num_movies_rated'])
        # Suprime solo los FutureWarnings específicos de Pandas
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            # Tu código que causa el FutureWarning
            user_df = pd.concat([user_df, row_df], ignore_index=True)

    user_df['user_vector'] = user_df['user_vector'].apply(lambda x: np.array_str(x))
    user_df['user_vector'] = user_df['user_vector'].apply(lambda x: x.replace('[', ' ').replace(']', ' ').strip().split())
    user_df['user_vector'] = user_df['user_vector'].apply(lambda x: np.asarray(x).astype(float))

    return user_df, idx_to_genre

def user_top_genre(user_df, idx_to_genre, userId):
    user_vec = user_df['user_vector'][user_df['userId'] == userId].values[0].copy()
    top_genre_indices = np.flip(np.argsort(user_vec))
    genre_list = []
    for i in top_genre_indices[:3]:
        genre_list.append(idx_to_genre[i])
    return genre_list

def recommender_popularity(df, user_id, n=30):
    user_df, idx_to_genre = InfoUser(df)
    genres = user_top_genre(user_df, idx_to_genre, user_id)
    recomendation = pd.DataFrame()
    for genre in genres:
        # Suprime solo los FutureWarnings específicos de Pandas
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            recomendation = pd.concat([recomendation, recommender_genre(df, genre, n*2)])
        recomendation.drop_duplicates(subset='id', keep='first', inplace=True)

    # Asegura que 'n' no sea mayor que la cantidad de películas disponibles
    n = min(n, len(recomendation))

    recomendation = recomendation.head(n)
    return recomendation

# Memory Based Recomendator:
def euclidean_similarity(ratings_dict, person1, person2):
    common_ranked_items = [itm for itm in ratings_dict[person1] if itm in ratings_dict[person2]]
    if len(common_ranked_items) == 0:
        return 0
    rankings = [(float(ratings_dict[person1][itm]), float(ratings_dict[person2][itm])) for itm in common_ranked_items]
    distance = [pow(rank[0] - rank[1], 2) for rank in rankings]
    similarity = 1 / (1 + np.sum(distance))
    return similarity

def pearson_similarity(ratings_dict, person1, person2):

	common_ranked_items = [itm for itm in ratings_dict[person1] if itm in ratings_dict[person2]]

	n = len(common_ranked_items)

	s1 = sum([ratings_dict[person1][item] for item in common_ranked_items])
	s2 = sum([ratings_dict[person2][item] for item in common_ranked_items])

	ss1 = sum([pow(ratings_dict[person1][item], 2) for item in common_ranked_items])
	ss2 = sum([pow(ratings_dict[person2][item], 2) for item in common_ranked_items])

	ps = sum([ratings_dict[person1][item] * ratings_dict[person2][item] for item in common_ranked_items])

	num = n * ps - (s1 * s2)

	den = math.sqrt((n * ss1 - math.pow(s1, 2)) * (n * ss2 - math.pow(s2, 2)))

	return (num / den) if den != 0 else 0

def recommend(ratings_dict, person, bound, movies_metadata, similarity=pearson_similarity):
    scores = [(similarity(ratings_dict, person, other), other) for other in ratings_dict if other != person]

    scores.sort()
    scores.reverse()
    scores = scores[0:bound]

    recomms = {}

    for sim, other in scores:
        ranked = ratings_dict[other]

        for itm in ranked:
            if itm not in ratings_dict[person]:
                weight = sim * ranked[itm]

                if itm in recomms:
                    s, weights = recomms[itm]
                    recomms[itm] = (s + sim, weights + [weight])
                else:
                    recomms[itm] = (sim, [weight])

    recommendations = {}

    for r in recomms:
        sim, item = recomms[r]
        recommendations[r] = np.sum(item) / sim

    # Get movie titles for top recommendations
    top_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:bound]
    top_recommendations_with_titles = []
    for movie_id, rating in top_recommendations:
        movie_title = movies_metadata.loc[movies_metadata['id'] == movie_id, 'title'].values
        if len(movie_title) > 0:
            top_recommendations_with_titles.append((movie_id, movie_title[0], rating))

    # Print recommendations as a table
    headers = ["Movie ID", "Title", "Rating"]
    print(tabulate(top_recommendations_with_titles, headers=headers, tablefmt="pretty"))

def movies_pearson_similarity(ratings_dict, movie1, movie2):
    ratings1 = []
    ratings2 = []

    for user in ratings_dict:
        if movie1 in ratings_dict[user] and movie2 in ratings_dict[user]:
            ratings1.append(ratings_dict[user][movie1])
            ratings2.append(ratings_dict[user][movie2])

    if not ratings1 or not ratings2:
        return 0  # Return zero similarity if no common ratings are found

    # Calculate Pearson correlation coefficient
    mean_rating1 = np.mean(ratings1)
    mean_rating2 = np.mean(ratings2)

    numerator = np.sum((x - mean_rating1) * (y - mean_rating2) for x, y in zip(ratings1, ratings2))
    denominator = np.sqrt(np.sum((x - mean_rating1)**2 for x in ratings1)) * np.sqrt(np.sum((y - mean_rating2)**2 for y in ratings2))

    if denominator == 0:
        return 0  # Return zero similarity if denominator is zero to avoid division by zero

    pearson_corr = numerator / denominator
    return pearson_corr

def recommend_similar_movies(ratings_dict, target_movie, movies_metadata, similarity=movies_pearson_similarity, num_recommendations=5):
    movie_similarities = {}

    print("Target Movie:", movies_metadata.loc[movies_metadata['id'] == target_movie, 'title'])

    for movie in ratings_dict:
        if movie != target_movie:
            similarity_score = similarity(ratings_dict, target_movie, movie)
            movie_similarities[movie] = similarity_score

    # Sort movies by their similarity scores
    similar_movies = sorted(movie_similarities.items(), key=lambda x: x[1], reverse=True)

    # Get movie titles for the top N most similar movies
    top_recommendations_with_titles = []
    for movie_id, score in similar_movies[:num_recommendations]:
        movie_title = movies_metadata.loc[movies_metadata['id'] == movie_id, 'title'].values
        if len(movie_title) > 0:
            top_recommendations_with_titles.append((movie_id, movie_title[0], round(score, 2)))

    # Define headers for the table
    headers = ["Movie ID", "Title", "Similarity"]

    # Print recommendations as a table
    print(tabulate(top_recommendations_with_titles, headers=headers, tablefmt="pretty"))

# Model Based Recomendator:
def convert_traintest_dataframe_forsurprise(training_dataframe, testing_dataframe):
    reader = Reader(rating_scale=(0, 5))
    trainset = Dataset.load_from_df(training_dataframe[['userId', 'movieId', 'rating']], reader)
    testset = Dataset.load_from_df(testing_dataframe[['userId', 'movieId', 'rating']], reader)
    trainset = trainset.construct_trainset(trainset.raw_ratings)
    testset = testset.construct_testset(testset.raw_ratings)
    return trainset, testset

def recommendation_model(model, trainset, testset):
  # Train the algorithm on the trainset, and predict ratings for the testset
  start_fit = time.time()
  model.fit(trainset)
  end_fit = time.time()
  fit_time = end_fit - start_fit

  # Predictions on testing set
  start_test = time.time()
  test_predictions = model.test(testset)
  end_test = time.time()
  test_time = end_test - start_test

  test_rmse = accuracy.rmse(test_predictions)
  test_mae = accuracy.mae(test_predictions)

  print('Tiempo de Entrenamiento: ', fit_time ,' segundos')
  print('Tiempo de Testeo: ', test_time,' segundos')
  return

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

def final_user_recomendation(user_id):
    # Usando el Modelo Hibrido
    _, modelSVD = dump.load("Modelos/modelSVD")
    movies_metadata = pd.read_parquet("input/movies_final.parquet")
    recommendation = generate_recommendation(user_id, modelSVD, movies_metadata)
    return recommendation


def final_movie_recomendation(movie):

    df = "ver de donde sacar el df"
    similarity_matrix = "ver de donde sacar la matriz"
    top5 = recommender(df, similarity_matrix, movie, 5)
    return top5

def final_genre_recomendation(genre):
    df_genre = pd.read_parquet("input/movies_with_genre.parquet")
    recommendations = recommender_genre(df_genre, genre, n=5)

    return recommendations