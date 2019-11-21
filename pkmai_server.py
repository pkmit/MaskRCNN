import os
import uuid
from flask import Flask, request, render_template, send_from_directory

from pkmai.pkmai_backbone import PKMAI_BACKBONE as serve

app = Flask(__name__, static_folder='processed')
ai = serve()

TEMP_FOLDER = os.path.join(os.getcwd(), 'temp', 'server')
PROCESSED_FOLDER = os.path.join(os.getcwd(), 'processed')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def prediction():

    post_img = request.files.get('ai_image')

    unique_name = uuid.uuid4()
    temp_save_path = os.path.join(TEMP_FOLDER, '{}.jpg'.format(unique_name))
    post_img.save(temp_save_path)
    ret_url = ai.prediction(temp_save_path)
    fn = ret_url.split('\\')
    fn = fn[len(fn) - 1]    
    # return "{}/{}".format(request.url_root, app.send_static_file(fn))    
    return '{}processed/{}'.format(request.url_root, fn)

app.run(debug=True)