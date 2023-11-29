from flask import Flask, request, jsonify
from inference import run_inference  # Assuming your inference method is in 'inference.py'

app = Flask(__name__)


# Endpoint to accept POST requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the POST request
        data = request.json

        # Assuming 'data' contains the input for your model

        # Call the method for inference from another file
        result = run_inference(data)  # Pass data to your inference function

        # Return the result as JSON
        return jsonify({'result': result}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
