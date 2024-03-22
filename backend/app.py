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
json_allcat_file_path = os.path.join(current_directory, "all_apps.json")

# Assuming your JSON data is stored in a file named 'init.json'
# with open(json_file_path, 'r') as file:
#   data = json.load(file)
#  episodes_df = pd.DataFrame(data['episodes'])
# reviews_df = pd.DataFrame(data['reviews'])

with open(json_allcat_file_path) as file:
    data = json.load(file)
    apps_df = pd.DataFrame(data)

app = Flask(__name__)
CORS(app)


# Sample search using json with pandas
def json_search(query):
    # basic jaccard on description and query
    words_set = set(query.lower().split())
    scores = []
    for ind in apps_df.index:
        desc_set = set(apps_df["description"][ind].lower().split())
        num = words_set.intersection(desc_set)
        den = words_set.union(desc_set)
        scores.append(len(num) / len(den))

    # argsort
    inds = sorted(range(len(scores)), key=scores.__getitem__)
    matches = []
    matches = apps_df.loc[inds]

    matches_filtered = matches[["title", "summary", "scoreText"]]
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


@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    return json_search(text)


if "DB_NAME" not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
