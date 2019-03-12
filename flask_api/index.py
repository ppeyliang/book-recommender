from flask import Flask, jsonify
from flask_cors import CORS
from flask import request
import json
import pickle
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import heapq
from operator import itemgetter
from surprise import Reader
from surprise import Dataset
from surprise import KNNWithZScore

app = Flask(__name__)
CORS(app)


@app.route('/recommend', methods=['POST'])
def recommend():
    # Append new ratings to DataFrame.
    my_dir = os.path.dirname(__file__)
    pickle_file_path = os.path.join(my_dir, 'cf.pkl')
    with open(pickle_file_path, "rb") as f:
        cf = pickle.load(f)
    df = cf[['user', 'ISBN', 'rating']]
    books = json.dumps(request.json)
    df_new = pd.read_json(books, orient='split')
    all_data = pd.concat([df, df_new]).drop_duplicates().reset_index(drop=True)

    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(all_data, reader)

    # Calculate similarity matrix.
    testSubject = 2033
    k = 20
    trainset = data.build_full_trainset()
    sim_options = {'name': 'cosine', 'user_based': True}
    model = KNNWithZScore(sim_options=sim_options)
    model.fit(trainset)
    simsMatrix = model.compute_similarities()

    # Get top k similar users to our test subject.
    testUserInnerID = trainset.to_inner_uid(testSubject)
    similarityRow = simsMatrix[testUserInnerID]
    similarUsers = []
    for innerID, score in enumerate(similarityRow):
        if (innerID != testUserInnerID):
            similarUsers.append((innerID, score))

    kNeighbors = heapq.nlargest(k, similarUsers, key=lambda t: t[1])

    # Get the items that similar users rated.
    results = defaultdict(list)
    for similarUser in kNeighbors:
        innerID = similarUser[0]
        userSimilarityScore = similarUser[1]
        theirRatings = trainset.ur[innerID]
        for item_inner_id, rating in theirRatings:
            results[item_inner_id].append((userSimilarityScore, rating))

    # Predict ratings for each item, weighted by user similarity.
    candidates = defaultdict(float)

    sim_rating = 0
    sim_sum = 0
    for item_inner_id, ratings in results.items():
        for similarity, rating in ratings:
            sim_rating += similarity * rating
            sim_sum += similarity
        pred_rating = sim_rating / sim_sum
        candidates[item_inner_id] = pred_rating

    # Build a dictionary of items the user has already read.
    read = {}
    for itemID, rating in trainset.ur[testUserInnerID]:
        read[itemID] = 1

    # Make recommendation.
    results = {'book_id': [], 'title': [],
               'author': [], 'year': [], 'image': []}

    for itemID, rating in sorted(candidates.items(), key=itemgetter(1), reverse=True):
        if not itemID in read:
            bookID = trainset.to_raw_iid(itemID)
            title = cf[cf['ISBN'] == bookID]['title'].unique()[0]
            author = cf[cf['ISBN'] == bookID]['author'].unique()[0]
            year = cf[cf['ISBN'] == bookID]['year'].unique()[0]
            image = cf[cf['ISBN'] == bookID]['image'].unique()[0]
            results['book_id'].append(bookID)
            results['title'].append(title)
            results['author'].append(author)
            results['year'].append(int(year))
            results['image'].append(image)

    output = pd.DataFrame(results).head(20)

    return output.to_json(orient='records').replace('\\', '')


if __name__ == '__main__':
    app.run()
