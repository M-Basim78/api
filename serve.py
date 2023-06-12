import numpy as np
from PIL import Image
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences 

def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension are correct")

    image = image.resize((299, 299))
    image = np.array(image)

    # For images that have 4 channels, convert them into 3 channels
    if image.shape[2] == 4:
        image = image[..., :3]

    image = np.expand_dims(image, axis=0)
    image = image / 127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)

        if word is None:
            break
        in_text += ' ' + word

        if word == 'end':
            break
    return in_text

# Load the tokenizer
tokenizer = None
with open('models/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_length = 51

# Load the model
model = None
model_path = 'models/new_model.h5'
model = load_model(model_path)

# Load the Xception model
xception_model = Xception(include_top=False, pooling="avg")

# img_paths = []  # Set this variable with your image paths
