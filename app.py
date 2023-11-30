from flask import Flask, request, jsonify, Response
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

app = Flask(__name__,  static_folder='frontend', static_url_path='')

frontend_directory = os.path.join(os.path.dirname(__file__), 'frontend')

@app.route('/predict', methods=['POST'])
def predict():
    # global movies_soup, movies_with_genre, ratings_sample, tags_matrix, similarity_matrix, modelKNN, modelSVD
    try:
        data = request.get_json()
        print(data)
        user_id = data.get('user_id', "")
        result_df = recommender_hybrid(ratings_sample, movies_soup, similarity_matrix, movies_with_genre, modelKNN, modelSVD, user_id, 5)
        # Obtener solo los títulos de las películas si result_df es un DataFrame
        if isinstance(result_df, pd.DataFrame):
            titles = result_df['title'].tolist()
            print(titles)
            return Response(json.dumps({'result': titles}), status=200, mimetype='application/json')
        else:
            return Response(json.dumps({'error': str(e)}), status=500, mimetype='application/json')

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

@app.route('/predictgenre', methods=['POST'])
def predict_genre():
    # global movies_with_genre
    try:
        data = request.get_json()
        print(data)
        genre_name = data.get('genre', "")
        result_df = recommender_genre(movies_with_genre, genre_name, 5)

        # Obtener solo los títulos de las películas si result_df es un DataFrame
        if isinstance(result_df, pd.DataFrame):
            titles = result_df['title'].tolist()
            print(titles)
            return jsonify({'result': titles}), 200
        else:
            return jsonify({'error': 'Result is not a DataFrame'}), 500

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

@app.route('/predictmovie', methods=['POST'])
def predict_movie():
    # global movies_with_genre
    try:
        data = request.get_json()
        print(data)
        movie_name = data.get('movie', "")
        result_df = recommender(movies_with_genre,similarity_matrix, movie_name, 5)

        # Obtener solo los títulos de las películas si result_df es un DataFrame
        if isinstance(result_df, pd.DataFrame):
            titles = result_df['title'].tolist()
            print(titles)
            return jsonify({'result': titles}), 200
        else:
            return jsonify({'error': 'Result is not a DataFrame'}), 500

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=True)
