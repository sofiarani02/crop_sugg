from flask import Flask, render_template
from crop import crop_prediction
from fertilizer import fertilizer_suggestion

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

app.register_blueprint(crop_prediction)
app.register_blueprint(fertilizer_suggestion)

if __name__ == '__main__':
    app.run(debug=True)
