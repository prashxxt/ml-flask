from flask import Flask, render_template, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load your trained model
model = joblib.load("Model.pkl")

# Initialize NLTK components outside the functions
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function for preprocessing text data
def clean(doc):
    doc = doc.replace("</br>", " ")
    doc = re.sub(r'[^\w\s]', '', doc)  # Remove punctuation
    doc = doc.lower()
    tokens = word_tokenize(doc)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    filtered_tokens = [word for word in lemmatized_tokens if word.lower() not in stop_words]
    return " ".join(filtered_tokens)

def map_label_to_sentiment(label):
    if label > 3:
        return 'Positive \U0001F604'
    elif label == 3:
        return 'Neutral \U0001F610'
    else:
        return 'Negative \U0001F641'

@app.route('/', methods=['GET', 'POST'])
def analyze():
    if request.method == 'GET':
        return render_template('home.html')
    elif request.method == 'POST':
        text = request.form['text']
        preprocessed_text = clean(text)
        prediction = model.predict([preprocessed_text])[0]
        result = map_label_to_sentiment(prediction)
        return render_template('result.html', text=text, result=result)

@app.route('/clean', methods=['POST'])
def clean_data():
    new_data = request.form.getlist('new_data')
    new_data_clean = [clean(doc) for doc in new_data]
    predictions = [map_label_to_sentiment(pred) for pred in model.predict(new_data_clean)]
    return jsonify(predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
