import nltk
from flask import Flask, render_template, request, redirect
import speech_recognition as sr
from nltk import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from fastpunct import FastPunct
from typing import List, Dict, Any

app = Flask(__name__)

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


@app.route("/", methods=["GET", "POST"])
def index():
    """
    :return: returns the mutated transcript to index.html so that statistics about the transcript can be displayed
    to the user.
    """
    sentiment_rating, show, char_count, word_count, letter_count = [0]*5
    transcript, enhanced_transcript, tokenized_transcript = [""]*3
    nouns, verbs, adjectives, adverbs, prepositions, determiners, pronouns, conjunctions = [[]]*8
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


def total_chars(transcript: str) -> int:
    """
    Sums the total amount of characters in the transcript.
    :param transcript: A string containing the contents of the transcribed audio file.
    :return: Returns the number of characters in the file.
    """
    counter = 0
    for i in transcript:
        counter += 1
    return counter


def total_letters(transcript: str) -> int:
    """
    Sums the total amount of non-space characters in the transcript.
    :param transcript: A string containing the contents of the transcribed audio file.
    :return: Returns the number of letters in the file.
    """
    counter = 0
    for i in transcript:
        if i != " ":
            counter += 1
    return counter


def total_words(transcript: str) -> int:
    """
    Returns the total number of words in the transcript.
    :param transcript: A string containing the contents of the transcribed audio file.
    :return: Returns the number of words in the file.
    """
    return len(transcript.split())


def parts_of_speech_categorize(postag_transcript: list) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    """
    Iterates through the tokenized, POS (Part Of Speech) assigned transcript
    :param postag_transcript: Transcript file tokenized and assigned a POS label by the nltk package.
    :return: Returns a tuple, each index of which contains a list of words that pertain to a given POS.
    """
    nouns, verbs, adjectives, adverbs, prepositions, determiners, pronouns, conjunctions = [[]]*8
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
    return nouns, verbs, adjectives, adverbs, prepositions, determiners, pronouns, conjunctions


def most_commonly_used_words(split_words: list) -> dict:
    """
    :param split_words: tokenized transcript.
    :return: Returns a dictionary with keys containing each word in the dictionary and values containing the amount of times it
    was used.
    """
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
