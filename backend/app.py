import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ["ROOT_PATH"] = os.path.abspath(os.path.join("..", os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
# json_file_path = os.path.join(current_directory, 'init.json')
json_allcat_file_path = os.path.join(current_directory, "data/all_apps.json")
json_allreviews_file_path = os.path.join(current_directory, "data/reviews.json")

# Assuming your JSON data is stored in a file named 'init.json'
# with open(json_file_path, 'r') as file:
#   data = json.load(file)
#  episodes_df = pd.DataFrame(data['episodes'])
# reviews_df = pd.DataFrame(data['reviews'])

with open(json_allcat_file_path) as file:
    data = json.load(file)
    apps_df = pd.DataFrame(data)
with open(json_allreviews_file_path) as file:
    data = json.load(file)
    rev_df = pd.DataFrame(data)

app = Flask(__name__)
CORS(app)


# Standardizes creating of word set
def clean(query):
    return set(query.lower().split())

#TODO: Preprocess the reviews
#       - avoid loading it in every single time
#       - consistent + faster

# Sample search using json with pandas
def json_search(query):
    words_set = clean(query)

    if len(words_set) == 0:
        raise ValueError("Query should not be empty")

    # basic jaccard on reviews and query
    reviewScores = {}
    totalScores = {}
    for ind in rev_df.index:
        if rev_df["thumbsUp"][ind] < 5:
            continue
        rev_set = clean(rev_df["text"][ind])
        num = words_set.intersection(rev_set)
        den = words_set.union(rev_set)
        score = len(num) / len(den)

        title = rev_df["appId"][ind]
        if title not in reviewScores.keys():
            reviewScores[title] = 0
        if title not in totalScores.keys():
            totalScores[title] = 0

        reviewScores[title] += score
        totalScores[title] += 1

    for key in reviewScores.keys():
        reviewScores[key] = reviewScores[key] / totalScores[key]

    # basic jaccard on description and query, also using review scores
    scores = []
    for ind in apps_df.index:
        desc_set = clean(apps_df["description"][ind])
        num = words_set.intersection(desc_set)
        den = words_set.union(desc_set)
        scores.append(len(num) / len(den) + reviewScores[apps_df["appId"][ind]])

    # argsort
    inds = sorted(range(len(scores)), key=scores.__getitem__)
    inds.reverse()
    matches = []
    matches = apps_df.loc[inds]

    matches_filtered = matches[["title", "summary", "scoreText", "appId", "icon"]]
    matches_filtered_json = matches_filtered.to_json(orient="records")
    return matches_filtered_json

#   matches = []
#  merged_df = pd.merge(episodes_df, reviews_df, left_on='id', right_on='id', how='inner')
# matches = merged_df[merged_df['title'].str.lower().str.contains(query.lower())]
# matches_filtered = matches[['title', 'descr', 'imdb_rating']]
# matches_filtered_json = matches_filtered.to_json(orient='records')
# return matches_filtered_json


@app.route("/")
def home():
    return render_template("base.html", title="sample html")


@app.route("/apps")
def episodes_search():
    text = request.args.get("title")
    return json_search(text)

@app.route("/inforeq")
def info_query():
    appId = request.args.get("appId")
    
    print("QUERYING INFO OF: " + appId)
    x = apps_df.loc[apps_df["appId"] == appId,:]
    return x.to_json(orient="records")


if "DB_NAME" not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
