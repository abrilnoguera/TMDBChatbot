from flask import Flask, request, jsonify
from funciones import *
import os
import pandas as pd


app = Flask(__name__,  static_folder='frontend', static_url_path='')

frontend_directory = os.path.join(os.path.dirname(__file__), 'frontend')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(data)
        user_id = data.get('user_id', "")
        result = final_user_recomendation(user_id)
        return jsonify({'result': result[0]}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predictgenre', methods=['POST'])
def predict_genre():
    try:
        data = request.get_json()
        print(data)
        movie_name = data.get('movie', "")
        result_df = final_genre_recomendation(movie_name)

        # Obtener solo los títulos de las películas si result_df es un DataFrame
        if isinstance(result_df, pd.DataFrame):
            titles = result_df['title'].tolist()
            print(titles)
            return jsonify({'result': titles}), 200
        else:
            return jsonify({'error': 'Result is not a DataFrame'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return app.send_static_file('index.html')


if __name__ == '__main__':
    app.run(debug=True)
