from flask import Flask, jsonify, request
import subprocess
import os

app = Flask(__name__)

@app.route('/retrain', methods=['POST'])
def retrain_model():
    try:
        # Define the path to the re_train.py script
        script_path = os.path.join(os.path.dirname(__file__), 're_train', 're_train.py')

        # Execute the re_train.py script
        # We use a blocking call here, so the API will wait for the retraining to complete
        result = subprocess.run(['python', script_path], capture_output=True, text=True, check=True)

        return jsonify({
            "status": "success",
            "message": "Model retraining initiated successfully.",
            "stdout": result.stdout,
            "stderr": result.stderr
        }), 200
    except subprocess.CalledProcessError as e:
        return jsonify({
            "status": "error",
            "message": "Error during model retraining.",
            "error_details": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr
        }), 500
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "An unexpected error occurred.",
            "error_details": str(e)
        }), 500

if __name__ == '__main__':
    # You can change the host and port as needed
    app.run(host='0.0.0.0', port=5001)
