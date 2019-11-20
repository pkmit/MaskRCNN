import os
import uuid
from flask import Flask, request

from pkmai.pkmai_backbone import PKMAI_BACKBONE as serve

app = Flask(__name__)

TEMP_FOLDER = os.path.join(os.getcwd(), 'temp', 'server')

@app.route('/')
def index():
    return "Welcome to PKM AI"

@app.route('/prediction', methods=['POST'])
def prediction():

    post_img = request.files.get('image')

    unique_name = uuid.uuid4()
    temp_save_path = os.path.join(TEMP_FOLDER, '{}.jpg'.format(unique_name))
    post_img.save(temp_save_path)          

    ai = serve()
    ai.prediction(temp_save_path)
    return 'Done prediction'

app.run(debug=True)