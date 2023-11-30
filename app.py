from flask import Flask, request, jsonify, Response
from funciones import *
import os
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import json
import unicodedata

# Carga de datos y modelos
movies_soup = pd.read_parquet('./input/movies_soup.parquet')
movies_with_genre = pd.read_parquet('./input/movies_with_genre.parquet')
ratings_sample = pd.read_parquet('./input/ratings_sample.parquet')
similarity_matrix = np.load('./input/similarity_matrix.npy')
_, modelKNN = dump.load('Modelos/modelKNN')
_, modelSVD = dump.load('Modelos/modelSVD')



# Inicialización de Flask
app = Flask(__name__,  static_folder='frontend', static_url_path='')

frontend_directory = os.path.join(os.path.dirname(__file__), 'frontend')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user = data.get('user_id', "")
        user_id = formatear_usuario(user)
        titles = recommender_hybrid(ratings_sample, movies_soup, similarity_matrix, movies_with_genre, modelKNN, modelSVD, user_id, 5)
        print('recomendacion: ', titles)
        return jsonify({'result': titles}), 200
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

@app.route('/predictgenre', methods=['POST'])
def predict_genre():
    try:
        data = request.get_json()
        print(data)
        genre_name = data.get('genre', "")
        genre_name = traducir_y_formatear_genero(genre_name)
        result_df = recommender_genre(movies_with_genre, genre_name, 5)

        # Obtener solo los títulos de las películas si result_df es un DataFrame
        if isinstance(result_df, pd.DataFrame):
            titles = result_df['title'].tolist()

            return jsonify({'result': titles}), 200
        else:
            return jsonify({'error': 'Result is not a DataFrame'}), 500

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

@app.route('/predictmovie', methods=['POST'])
def predict_movie():
    try:
        data = request.get_json()
        print(data)
        movie_name = data.get('movie', "")
        movie_name = movie_name.lower().title()
        result_df = recommender(movies_soup,similarity_matrix, movie_name, 5)

        # Obtener solo los títulos de las películas si result_df es un DataFrame
        if isinstance(result_df, pd.DataFrame):
            titles = result_df['title'].tolist()

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
