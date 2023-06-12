from flask import Flask, render_template, request, jsonify
from serve import extract_features, generate_desc, word_for_id
from PIL import Image
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences 

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/api/process')
def process():
    print("Process")

    return "OK"

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    model = load_model('models/new_model.h5')
    with open('models/tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    xception_model = Xception(include_top=False, pooling="avg")
    # Extract the image file from the request
    file = request.files['file']

    # Save the image file to a local directory
    filename = 'temp.jpg'
    file.save(filename)

    # Perform the image processing and caption generation
    # ... your existing code goes here ...
    photo = extract_features(filename, xception_model)
    max_length= 51
    img = Image.open(filename)
    description = generate_desc(model, tokenizer, photo, max_length)
    caption = description.strip()

    # Return the caption as a JSON response
    response = {'caption': caption}
    return jsonify(response)

def load_model_and_tokenizer():
    global model, tokenizer, xception_model

    # Load the model and tokenizer
    


if __name__ == '__main__':
    app.run(port = 8888, host='0.0.0.0')
    
