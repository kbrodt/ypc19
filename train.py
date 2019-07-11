import argparse
from collections import defaultdict, Counter
import copy as c
from datetime import datetime
import random
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import tqdm
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import BM25Recommender, ItemItemRecommender


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--data_path',  default=Path('Blitz'))
    
    arg('--smoothing',   type=float, default=0.99)
    arg('--bm25-k',      type=int, default=600)
    arg('--i2i-k',       type=int, default=200)
    arg('--als-factors', type=int, default=512)
    arg('--als-iters',   type=int, default=15)
    arg('--top-k',       type=int, default=400)
    arg('--rating',      type=float, default=50.0)
    
    arg('--seed', type=int, default=314159)
    
    return parser.parse_args()


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    

def read_data(root):
    test_users = set(pd.read_csv(root / 'test_users.zip').user_id)

    predictions = {user_uid: [] for user_uid in test_users}

    train_clicks = pd.read_csv(root/ 'train_clicks.zip')
    train_likes = pd.read_csv(root/ 'train_likes.zip')
    train_shares = pd.read_csv(root/ 'train_shares.zip')

    train_clicks.drop_duplicates(subset=['user_id', 'picture_id', 'day'], inplace=True)
    train_likes.drop_duplicates(subset=['user_id', 'picture_id', 'day'], inplace=True)
    train_shares.drop_duplicates(subset=['user_id', 'picture_id', 'day'], inplace=True)

    train_clicks['day'] = pd.to_datetime(train_clicks['day'])
    train_likes['day'] = pd.to_datetime(train_likes['day'])
    train_shares['day'] = pd.to_datetime(train_shares['day'])

    train_clicks.sort_values('day', inplace=True)
    train_likes.sort_values('day', inplace=True)
    train_shares.sort_values('day', inplace=True)

    picture_descriptions = pd.read_csv(root/ 'descriptions.zip')
    themes = pd.read_csv(root / 'themes.zip')

    user_profiles = pd.read_csv(root / 'user_information.zip')

    user_profiles.drop_duplicates(subset=['user_id', 'day', 'embedding'], inplace=True)
    user_profiles.drop_duplicates(subset=['user_id', 'day'], inplace=True)
    user_profiles['day'] = pd.to_datetime(user_profiles['day'])
    user_profiles.sort_values('day', inplace=True)

    catalogue = (set(themes.picture_id)
                 | set(picture_descriptions.picture_id)
                 | set(train_shares.picture_id.values)
                 | set(train_likes.picture_id.values)
                 | set(train_clicks.picture_id.values))
    catalogue = pd.DataFrame(catalogue, columns=['picture_id'])
    catalogue = pd.merge(catalogue, picture_descriptions, how='left', on='picture_id')
    catalogue = pd.merge(catalogue, themes, how='left', on='picture_id')

    transactions = train_clicks
    transactions_with_catalogue = transactions.join(catalogue.set_index('picture_id'),
                                                    how='inner',
                                                    on='picture_id',
                                                    sort='day')

    ratings = train_likes
    bookmarks = train_shares

    user_consumed_pictures = defaultdict(set)
    with tqdm.tqdm(transactions_with_catalogue.loc[:, ['user_id', 'picture_id']].values,
                   desc='[ User consumed pictures.. ]') as pbar:
        for user_uid, element_uid in pbar:
            user_consumed_pictures[user_uid].add(element_uid)

    with tqdm.tqdm(train_likes.loc[:, ['user_id', 'picture_id']].values,
                   desc='[ User consumed pictures.. ]') as pbar:
        for user_uid, element_uid in pbar:
            user_consumed_pictures[user_uid].add(element_uid)

    return test_users, predictions, transactions_with_catalogue, ratings, bookmarks, user_consumed_pictures


def evaluate_model(data, model, N=200):
    test_users, predictions, picture_score_by_user, top_pictures, users_bookmarked_pictures, user_consumed_pictures = data
    picture_score_by_user['user_id'] = picture_score_by_user['user_id'].astype('category')
    picture_score_by_user['picture_id'] = picture_score_by_user['picture_id'].astype('category')
    ratings_matrix = sp.coo_matrix(
        (picture_score_by_user['rating'].values.astype(np.float64) + 1,
         (
             picture_score_by_user['picture_id'].cat.codes.copy(),
             picture_score_by_user['user_id'].cat.codes.copy()
         )
         )
    )

    ratings_matrix = ratings_matrix.tocsr()
    ratings_matrix_T = ratings_matrix.T.tocsr()

    model.fit(ratings_matrix)

    user_uid_to_cat = dict(zip(
        picture_score_by_user['user_id'].cat.categories,
        range(len(picture_score_by_user['user_id'].cat.categories))
    ))

    element_uid_to_cat = dict(zip(
        picture_score_by_user['picture_id'].cat.categories,
        range(len(picture_score_by_user['picture_id'].cat.categories))
    ))

    filtered_elements_cat = {k: [element_uid_to_cat.get(x, None) for x in v]
                             for k, v in user_consumed_pictures.items()}

    model_predictions = defaultdict(list)
    with tqdm.tqdm(test_users, desc='[ Predicting.. ]') as pbar:
        for user_uid in pbar:
            try:
                user_cat = user_uid_to_cat[user_uid]
            except LookupError:
                continue

            recs = model.recommend(
                user_cat,
                ratings_matrix_T,
                N=N,
                filter_already_liked_items=True,
                filter_items=filtered_elements_cat.get(user_uid, set())
            )

            model_predictions[user_uid] = [int(picture_score_by_user['picture_id'].cat.categories[i])
                                           for i, _ in recs]

    with tqdm.tqdm(test_users, desc='[ Filtering.. ]') as pbar:
        for user_uid in pbar:
            for picture in model_predictions[user_uid]:
                if len(predictions[user_uid]) == N:
                    break
                if picture in users_bookmarked_pictures[user_uid] and picture not in user_consumed_pictures[user_uid]:
                    predictions[user_uid].append(picture)
                    user_consumed_pictures[user_uid].add(picture)

            for picture in users_bookmarked_pictures[user_uid]:
                if len(predictions[user_uid]) == N:
                    break
                if picture not in user_consumed_pictures[user_uid]:
                    predictions[user_uid].append(picture)
                    user_consumed_pictures[user_uid].add(picture)

            for picture in model_predictions[user_uid]:
                if len(predictions[user_uid]) == N:
                    break
                if picture not in user_consumed_pictures[user_uid]:
                    predictions[user_uid].append(picture)
                    user_consumed_pictures[user_uid].add(picture)

            for picture in top_pictures:
                if len(predictions[user_uid]) == N:
                    break
                if picture not in user_consumed_pictures[user_uid]:
                    predictions[user_uid].append(picture)
                    user_consumed_pictures[user_uid].add(picture)

    return predictions


def evaluate(data, smoothing, models_list, r=50.0, N=200, EPS=0.00001):
    test_users, predictions, transactions_with_catalogue, ratings, bookmarks, user_consumed_pictures = data

    max_date = max(transactions_with_catalogue['day'])
    transactions_with_catalogue['rating'] = pd.Series(np.full(len(transactions_with_catalogue), r),
                                                      index=transactions_with_catalogue.index)
    new_ratings = []
    with tqdm.tqdm(transactions_with_catalogue.iterrows(),
                   total=len(transactions_with_catalogue),
                   desc='[ Creating ratings for transactions.. ]') as pbar:
        for _, row in pbar:
            rating, date = row.rating, row.day
            new_ratings.append(rating * smoothing ** (max_date - date).days + EPS)
    transactions_with_catalogue['rating'] = pd.Series(new_ratings, index=transactions_with_catalogue.index)

    new_ratings = []
    with tqdm.tqdm(ratings.iterrows(),
                   total=len(ratings),
                   desc='[ Creating ratings for likes.. ]') as pbar:
        for _, row in pbar:
            date = row.day
            new_ratings.append(r * smoothing ** (max_date - date).days)
    ratings['rating'] = pd.Series(new_ratings, index=ratings.index)

    picture_score_by_user = pd.concat([ratings.loc[:, ['picture_id', 'user_id', 'rating']],
                                       transactions_with_catalogue.loc[:, ['picture_id', 'user_id', 'rating']]],
                                      ignore_index=True).groupby(
        ['picture_id', 'user_id'], as_index=False).sum().sort_values(by=['user_id'])

    movies_scores = picture_score_by_user.loc[:, ['picture_id', 'rating']].groupby(
        'picture_id', as_index=False).sum().sort_values(by=['rating'], ascending=False)
    top_pictures = list(movies_scores['picture_id'])[:N]

    users_bookmarked_pictures = defaultdict(list)
    bookmarks_with_scores = bookmarks.join(movies_scores.set_index('picture_id'), how='inner',
                                           on='picture_id', sort='picture_id')
    with tqdm.tqdm(bookmarks_with_scores.iterrows(),
                   total=len(bookmarks_with_scores),
                   desc='[ Creating bookmarks.. ]') as pbar:
        for _, row in pbar:
            users_bookmarked_pictures[row['user_id']].append((row['picture_id'], row['rating']))

    with tqdm.tqdm(users_bookmarked_pictures, desc='[ Creating bookmarks.. ]') as pbar:
        for user_uid in pbar:
            users_bookmarked_pictures[user_uid] = list(map(lambda x: int(x[0]),
                                                           sorted(users_bookmarked_pictures[user_uid],
                                                                  key=lambda x: x[1], reverse=True)))

    result = []
    data = test_users, predictions, picture_score_by_user, top_pictures, users_bookmarked_pictures, user_consumed_pictures
    for model in models_list:
        result.append(evaluate_model(c.deepcopy(data), model, N=N))

    return result


def mix_solutions(result, rates, pictures_num_to_leave):
    scores = dict()
    for user_uid in result[0]:
        scores[user_uid] = Counter()

    with tqdm.tqdm(result, desc='[ Mixing.. ]') as pbar:
        for i, prediction in enumerate(pbar):
            for user_uid in prediction:
                for pos, picture in enumerate(prediction[user_uid]):
                    scores[user_uid][picture] += rates[i] * 0.99 ** pos

    final_predictions = dict()
    with tqdm.tqdm(scores) as pbar:
        for user_uid in pbar:
            final_predictions[user_uid] = list(map(lambda x: x[0],
                                                   sorted(scores[user_uid].items(),
                                                          key=lambda x: x[1], reverse=True)))[:pictures_num_to_leave]

    return final_predictions


def main():
    args = get_args()
    
    set_seeds(args.seed)
    
    data_path = args.data_path
    
    data = read_data(root=data_path)
    result = evaluate(data=c.deepcopy(data),
                      smoothing=args.smoothing, 
                      models_list=[
                          BM25Recommender(K=args.bm25_k),
                          ItemItemRecommender(K=args.i2i_k),
                          AlternatingLeastSquares(factors=args.als_factors, iterations=args.als_iters),
                      ],
                      r=args.rating,
                      N=args.top_k
                     )

    RATES = [5.5, 2, 6]
    TOP_K = 100

    predictions = mix_solutions(result=result, rates=RATES, pictures_num_to_leave=TOP_K)

    test_users = pd.DataFrame.from_dict(predictions).T.reset_index()
    test_users.rename({'index': 'user_id'}, inplace=True, axis=1)
    test_users.sort_values('user_id', inplace=True)
    test_users['predictions'] = test_users[list(range(TOP_K))].apply(lambda x: ' '.join(map(str, x)), axis=1)
    test_users[['user_id', 'predictions']].to_csv('sub.csv', index=False)


if __name__ == '__main__':
    main()