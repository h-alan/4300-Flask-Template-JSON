import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import math
import numpy as np

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

with open(json_allcat_file_path, encoding="utf-8") as file:
    data = json.load(file)
    apps_df = pd.DataFrame(data).drop_duplicates(subset=['appId'])
with open(json_allreviews_file_path, encoding="utf-8") as file:
    data = json.load(file)
    rev_df = pd.DataFrame(data).drop_duplicates(subset=['appId'])

app = Flask(__name__)
CORS(app)


# Standardizes creating of word set
def clean(query):
    return set(query.lower().split())


def tokenize_input(input):
    return input.replace(".", " ").replace(",", " ").replace("?", " ").replace("!", " ").replace("-", " ").split()

def build_tf_inv_idx(df, key):
    output = {}

    for ind in df.index:
        # removing punctuation
        # replace is faster than translate
        desc = tokenize_input(df[key][ind].lower())

        # building tf inv_idx dict
        counts = {}
        for token in desc:
            counts[token] = counts.get(token, 0) + 1
        for token in counts:
            if token in output:
                output[token].append((ind, counts[token]))
            else:
                output[token] = [(ind, counts[token])]

    return output


def compute_idf(inv_idx, n_docs, min_df, max_df_ratio):
    res = {}

    for token in inv_idx:
        docs = inv_idx[token]

        # filter out tokens too frequent or not frequent enough
        if len(docs) < min_df:
            continue
        ratio = len(docs) / n_docs
        if ratio > max_df_ratio:
            continue
        else:
            res[token] = math.log2(n_docs / (1 + len(docs)))

    return res


def compute_norms(inv_idx, idf, n_docs):
    res = np.zeros(shape=n_docs)

    for token in inv_idx:
        for doc, freq in inv_idx[token]:
            if token in idf:
                res[doc] += (freq * idf[token]) ** 2

    for i in range(n_docs):
        res[i] = math.sqrt(res[i])

    return res


# precomputing before query is input
desc_inv_idx = build_tf_inv_idx(apps_df, "description")
desc_idf_dict = compute_idf(desc_inv_idx, apps_df.size, 0, 0.9)
desc_norms = compute_norms(desc_inv_idx, desc_idf_dict, apps_df.size)

# this takes forever to finish
""" precomputing for each review
rev_dict = {}
for ind in rev_df.index:
    if rev_df["thumbsUp"][ind] < 5:
        continue
    rev_dict[ind] = {}
    rev_dict[ind]['inv_idx'] = build_tf_inv_idx(rev_df)
    rev_dict[ind]['idf'] = compute_idf(rev_dict[ind]['inv_idx'], rev_df.size, 5, 0.95)
    rev_dict[ind]['norms'] = compute_norms(rev_dict[ind]['inv_idx'], rev_dict[ind]['idf'], rev_df.size)
"""


def jaccard_similarity(words_set):
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


def jaccard_reviews(words_set):
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

    return reviewScores

def compute_dot_scores(query_word_counts, inv_idx, idf):
    doc_scores = {}

    for token in query_word_counts:
        if token in inv_idx and token in idf:
            for doc, freq in inv_idx[token]:
                doc_scores[doc] = (
                    doc_scores.get(doc, 0)
                    + freq * idf[token] * query_word_counts[token] * idf[token]
                )

    return doc_scores


def compute_cosine_sim(query, inv_idx, idf, doc_norms):
    q_count = {}
    q_norm = 0
    for token in query:
        q_count[token] = q_count.get(token, 0) + 1
    for token in q_count:
        if token in idf:
            q_norm += (q_count[token] * idf[token]) ** 2
    q_norm = math.sqrt(q_norm)

    res = {}
    doc_scores = compute_dot_scores(q_count, inv_idx, idf)
    for doc in doc_scores:
        res[doc] = doc_scores[doc] / (q_norm * doc_norms[doc])

    return res


# Returns sorted list
def cosine_similarity(query, desc_idx, desc_idf, desc_doc_norms, rev_dict):
    tokens = tokenize_input(query.lower())
    desc_sim = compute_cosine_sim(tokens, desc_idx, desc_idf, desc_doc_norms)

    # computing average review cosine score for each app
    """
    app_rev_score = {}
    app_rev_count = {}
    for rev in rev_dict:
        score = compute_cosine_sim(rev_dict[rev]['inv_idx'], rev_dict[rev]['idf'], rev_dict[rev]['norms'])
        title = rev_df["appId"][ind]
        app_rev_score[title] = app_rev_score.get(title, 0) + score
        app_rev_count[title] = app_rev_count.get(title, 0) + 1
    for app in app_rev_score:
       app_rev_score[app] = app_rev_score[app] / app_rev_count[app]
    """

    words_set = clean(query)
    app_rev_score = jaccard_reviews(words_set)

    combined = {}
    for key in desc_sim:
        if apps_df["appId"][key] not in app_rev_score:
           rev_score = 0
        else:
           score = app_rev_score[apps_df["appId"][key]]
        combined[key] = desc_sim[key] + score

    # switch this to combined once reviews get added
    inds = sorted(combined, key=combined.get, reverse=True)[0:10]
    matches = apps_df.loc[inds]

    #for w in sorted(desc_sim, key=desc_sim.get, reverse=True):
    #    print(apps_df["title"][w], desc_sim[w])

    matches_filtered = matches[
        ["title", "summary", "scoreText", "appId", "icon", "url", "price", "offersIAP", "score"]
    ]
    return matches_filtered

# apply filter
# though honestly, I think it's smarter to just do this on the frontend and not bother
# with this in the backend
def apply_filter(results, price, iap, score):
    filtered_results = results.query("price <= @price & (offersIAP | @iap) & score >= @score")
    return filtered_results

# Search using json with pandas
# filter values included are: max price of app (0-100), minimum rating (0-5) and
# whether in app purchases are allowed (boolean)
def json_search(text, price, iap, score):

    ranks = cosine_similarity(
        text, desc_inv_idx, desc_idf_dict, desc_norms, rev_dict={}
    )
    filtered_ranks = apply_filter(ranks, price, iap, score)
    return filtered_ranks.to_json(orient="records")
    


@app.route("/")
def home():
    return render_template("base.html", title="sample html")


@app.route("/apps")
def episodes_search():
    text = request.args.get("title")

    # empty query is allowed, we just return nothing
    if len(text) == 0:
        empty_data = json.loads("{}")
        return empty_data
    score = 0
    try:
        score = float(request.args.get("min_rating"))
    except:
        print("Unexpected error reading rating: " + request.args.get("min_rating"))
        score = 0
    price = 100
    try:
        price = float(request.args.get("max_price"))
    except:
        print("Unexpected error reading price: " + request.args.get("max_price"))
        price = 0.0
    iap = True
    try:
        iap = bool(request.args.get("iap"))
    except:
        print("Unexpected error reading iap: " + request.args.get("iap"))
        iap = True

    print(
        f"""SEARCHING: {text} 
\t min_rating: {score}
\t max_price: {price}
\t iap: {iap}""")
    return json_search(text, price, iap, score)


@app.route("/inforeq")
def info_query():
    appId = request.args.get("appId")

    print("QUERYING INFO OF: " + appId)
    x = apps_df.loc[apps_df["appId"] == appId, :]
    return x.to_json(orient="records")


@app.route("/rel-feed")
def query_improvement():
    
    # CONSTANTS/HYPERPARAMETERS:
    ROCCHIO_A = 0.8
    ROCCHIO_B = 0.3
    ROCCHIO_C = 0.4
    
    iteration_num = int(request.args.get("iter"))
    print(f"ROCCHIO ITERATION: {iteration_num}")
    rels = json.loads(request.args.get('rel'))[0]
    print(f"RELEVANT: {rels}")
    irrels = json.loads(request.args.get('irrel'))[0]
    print(f"IRRELEVANT: {irrels}")
    # rel, irrel are 2d arrays that contain all previous rocchio results / processes
    # do rocchio stuff
    # pls return some new rankings in similar way to JSON_search, or similar format to above
    
    query_str = request.args.get("title")
    query = tokenize_input(query_str.lower())
    
    # empty query is allowed, we just return nothing
    if len(query) == 0:
        empty_data = json.loads('{}')
        return empty_data
    
    for rel in rels:
        query+=tokenize_input(apps_df[apps_df['appId']==rel]['description'].tolist()[0].lower())

    return cosine_similarity(query, desc_inv_idx, desc_idf_dict, desc_norms, rev_df).to_json(orient="records")


if "DB_NAME" not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
