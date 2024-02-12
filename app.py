from flask import Flask, render_template, request
import random
import json
import torch
import os  # Add this line to import the 'os' module
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

app = Flask(__name__)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get the directory of the current script
current_directory = os.path.dirname(os.path.realpath(__file__))

# Construct the full path to intents.json
text_file_path = os.path.join(current_directory, "intents.json")

with open(text_file_path, 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
# app = Flask(__name__)
bot_name = "Sam"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def chat():
    user_message = request.form['user_message']

    sentence = tokenize(user_message)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                bot_response = random.choice(intent['responses'])
    else:
        bot_response = "I do not understand..."

    return render_template('index.html', user_message=user_message, bot_response=bot_response)


if __name__ == '__main__':
    app.run()