import nltk
from flask import Flask, render_template, request, redirect
import speech_recognition as sr
from nltk import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from fastpunct import FastPunct

app = Flask(__name__)

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


@app.route("/", methods=["GET", "POST"])
def index():
    sentiment_rating = 0
    show = 0
    char_count = 0
    word_count = 0
    letter_count = 0
    transcript = ""
    enhanced_transcript = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            recognizer = sr.Recognizer()
            audioFile = sr.AudioFile(file)
            with audioFile as source:
                data = recognizer.record(source)
            transcript = recognizer.recognize_google(data, key=None)
            char_count = total_chars(transcript)
            letter_count = total_letters(transcript)
            word_count = total_words(transcript)
            fastpunct = FastPunct()
            enhanced_transcript = fastpunct.punct(transcript)

            vader = SentimentIntensityAnalyzer()
            sentiment_rating = vader.polarity_scores(enhanced_transcript)
            th = word_tokenize(transcript)
            show = nltk.pos_tag(th)
            x = parts_of_speech_categorize(show)
            nouns = x[0]
            verbs = x[1]
            adjectives = x[2]
            adverbs = x[3]
            prepositions = x[4]
            determiners = x[5]
            pronouns = x[6]
            conjunctions = x[7]

    return render_template('index.html', transcript=transcript,
                           char_count=char_count, word_count=word_count,
                           letter_count=letter_count,
                           enhanced_transcript=enhanced_transcript,
                           sentiment_rating=sentiment_rating)


def total_chars(var):
    counter = 0
    for i in var:
        counter += 1
    return counter


def total_letters(var):
    counter = 0
    for i in var:
        if i != " ":
            counter += 1
    return counter


def total_words(var):
    return len(var.split())



if __name__ == "__main__":
    app.run(debug=True, threaded=True)
