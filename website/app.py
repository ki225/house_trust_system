from flask import Flask, request, jsonify

app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=True)
