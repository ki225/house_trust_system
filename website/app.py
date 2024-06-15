from flask import Flask, request, jsonify
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
FIXED_FILENAME = 'crack.png'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.get_json()
    user_message = data['message']
    # 把下面的東西改成大型語言模型統整的結果
    reply = f"I received your message: {user_message}"
    return jsonify({'reply': reply})

@app.route('/submit', methods=['POST'])
def upload_file():
    city = request.form.get('City')
    fault = request.form.get('Fault')
    soil_liquefaction = request.form.get('Soil_Liquefaction')
    land_subsidence = request.form.get('Land_Subsidence')
    material = request.form.get('Material')
    floor = request.form.get('Floor')

    if 'image' not in request.files:
        return 'No file part'
    file = request.files['image']

    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], FIXED_FILENAME)
        file.save(file_path)
        return app.send_static_file('index.html')
    return 'File type not allowed'

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
