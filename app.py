import nltk
from flask import Flask, render_template, request, redirect
import speech_recognition as sr
from nltk import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from fastpunct import FastPunct
from typing import List, Dict

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
    tech_text = ""
    tokenized_transcript = ""
    nouns, verbs, adjectives, adverbs, prepositions, determiners, pronouns, \
    conjunctions = [], [], [], [], [], [], [], []
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
            fast_punct = FastPunct()
            enhanced_transcript = fast_punct.punct(transcript)

            vader = SentimentIntensityAnalyzer()
            sentiment_rating = vader.polarity_scores(enhanced_transcript)
            tokenized_transcript = word_tokenize(transcript)
            show = nltk.pos_tag(tokenized_transcript)
            x = parts_of_speech_categorize(show)
            nouns, verbs, adjectives, adverbs, prepositions, \
            determiners = x[0], x[1], x[2], x[3], x[4], x[5]
            pronouns = x[6]
            conjunctions = x[7]

    return render_template('index.html',
                           transcript=transcript,
                           char_count=char_count,
                           word_count=word_count,
                           letter_count=letter_count,
                           enhanced_transcript=enhanced_transcript,
                           sentiment_rating=sentiment_rating,
                           nouns=nouns,
                           verbs=verbs,
                           adjectives=adjectives,
                           adverbs=adverbs,
                           prepositions=prepositions,
                           determiners=determiners,
                           pronouns=pronouns,
                           conjunctions=conjunctions,
                           common_words=most_commonly_used_words(
                               tokenized_transcript))


def total_chars(transcript):
    counter = 0
    for i in transcript:
        counter += 1
    return counter


def total_letters(transcript):
    counter = 0
    for i in transcript:
        if i != " ":
            counter += 1
    return counter


def total_words(transcript):
    return len(transcript.split())


def parts_of_speech_categorize(postag_transcript):
    nouns, verbs, adjectives, adverbs, prepositions, determiners, pronouns, \
    conjunctions = [], [], [], [], [], [], [], []
    for item in postag_transcript:
        word = item[0]
        POS = item[1]
        if POS in ['NN', 'NNS', 'NNP', 'NNPS']:
            nouns.append(word)
        elif POS in ['VB', 'VBG', 'VBN', 'VBP', 'VBD', 'VBZ']:
            verbs.append(word)
        elif POS in ['JJ', 'JJR', 'JJS']:
            adjectives.append(word)
        elif POS == 'IN':
            prepositions.append(word)
        elif POS == 'DT':
            determiners.append(word)
        elif POS == 'CC':
            conjunctions.append(word)
        elif POS in ['PRP', 'PRP$']:
            pronouns.append(word)
        elif POS in ['RB', 'RBR', 'RBS']:
            adverbs.append(word)
    return nouns, verbs, adjectives, adverbs, prepositions, determiners, \
           pronouns, conjunctions


def most_commonly_used_words(split_words):
    most_common = {}
    for word in split_words:
        if word not in most_common:
            most_common[word] = 1
        else:
            most_common[word] += 1
    most_common_lite = {}
    for key, value in most_common.items():
        most_common_lite[value] = key

    sorted_by_val = sorted(most_common.values())
    fully_sorted = {}
    for i in sorted_by_val:
        for k in most_common_lite.keys():
            if most_common_lite[k] == i:
                fully_sorted[k] = most_common_lite[k]
                break
    return fully_sorted


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
