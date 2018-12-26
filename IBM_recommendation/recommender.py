import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import project_tests as t
import pickle
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
%load_ext autoreload
%autoreload 2
from utils import tokenizer
from collections import Counter
from scipy import linalg
import sys # can use sys to take command line arguments
import recommender_functions_utils as rf

class Recommender():
    '''
    This Recommender uses FunkSVD, rank-based, mix(FunkSVD+rank-based) model for users,
    as well as content-base method for movies to make recommnedation. 
    Also uses either FunkSVD or mix method to make prediction on review rating for particular user and movie. 
    '''
    def __init__(self, method='user-base', _idtype='user', latent_features=12, learning_rate=0.0001, iters=500, rec_num=5, print_every=10):
        '''
        This function initialize the class
        INPUT:
        method - (str) 'user-base': user similarity, 'content-base': item NLP content, 'rank-base': most popularity, 'rank-base-content': rank-base with filtered NLP
        _idtype - (str) 'user' or 'item'
                        'user' only compatible with 'user-base'/'rank-base'/'rank-base-content' methods
                        'item' only compatible with 'content-base' methods

        latent_features - (int) the number of latent features used in 'mix'/funk-SVD' methods
        learning_rate - (float) the learning rate in 'mix'/funk-SVD' methods
        iters - (int) the number of iterations in 'mix'/funk-SVD' methods
        rec_num - (int) number of recommendations for output
        print_every - (int) interval between prints in 'mix'/funk-SVD' methods

        OUTPUT:
        None - stores the following as attributes:
        method - (str) 'mix': funk-SVD+rank-base, 'funk-SVD', 'rank-base', 'content-base'
        _idtype - (str) 'user' or 'movie'
                        'user' only compatible with 'mix'/'funk-SVD'/'rank-base' methods
                        'movie' only compatible with 'content-base' methods

        latent_features - (int) the number of latent features used in 'mix'/funk-SVD' methods
        learning_rate - (float) the learning rate in 'mix'/funk-SVD' methods
        iters - (int) the number of iterations in 'mix'/funk-SVD' methods
        rec_num - (int) number of recommendations for output
        print_every - (int) interval between prints in 'mix'/funk-SVD' methods
        review - review dataframe
        movies - movies dataframe
        train_user_item - reviews with fewer columns
        movie_ids_list - list of movies in movies dataframe
        user_ids_series - list of users in review datafarame
        movie_ids_series - list of movies in review dataframe
        dot_prod_movies_df - movie similarity (dot product) of categories in dataframe
        ranked_movies - review-movie join dataframe ordered by abg rating, date, with more than 4 review
        user_mat - user * latent_feature matrix from SVD/mix
        movie_mat -  latent_feature * movie matrix from SVD/mix
        '''

        self.method = method
        self._idtype = _idtype
        self.latent_features = latent_features
        self.learning_rate = learning_rate
        self.iters = iters
        self.rec_num = rec_num
        self.print_every = print_every

        if((self._idtype == 'user') and (self.method not in ['user-base', 'rank-base', 'rank-base-content'])):
            print("For 'user' idtype must initialize with one of 'user-base'/'rank-base'/'rank-base-content' methods")
            print("By default set method to 'user-base'")
            self.method = 'user-base'

        if((self._idtype == 'item') and (self.method not in ['content-base'])):
            print("For 'item' idtype must initialize with 'content-base'")
            print("By default set method to 'content-base'")
            self.method = 'content-base'
 
    def fit(self, df_content, df):
        '''
        fit the recommender to your dataset and also have this save the results
        to pull from when you need to make predictions
        '''
        self.df = df
        self.df_content = df_content

        self.train_user_item = self.reviews[['user_id', 'movie_id', 'rating', 'timestamp']]
        movie_content = np.array(self.movies.iloc[:,4:])
        print('Fitting .......')
        print('Method: {} id_type: {}'.format(self.method, self._idtype))
        print()


        if(self._idtype == 'user'):
            if((self.method == 'rank-base') or (self.method == 'rank-base-content')):
                self.ranked_df = rf.get_rank_data(df=self.df)

            if((self.method == 'user-base')):
                self.user_item, self.user_item_matrix = rf.create_user_item_matrix(df=self.df)

                # user/movie matrix with rating in bulk
                self.train_data_df = self.train_user_item.groupby(['user_id', 'movie_id'])['rating'].max().unstack()
                self.user_ids_series = np.array(self.train_data_df.index)
                #print(8 in self.user_ids_series)
                self.movie_ids_series = np.array(self.train_data_df.columns)
                self.train_data_np = np.array(self.train_data_df)
                self.user_mat, self.movie_mat = rf.FunkSVD(self.train_data_np, self.latent_features, self.learning_rate, self.iters, self.print_every)

        if(self._idtype == 'movie'):
            if(self.method == 'content-base'):
                dot_prod_movies = movie_content.dot(np.transpose(movie_content))
                self.movie_ids_list = self.movies.movie_id.values
                self.dot_prod_movies_df = pd.DataFrame(dot_prod_movies, columns=self.movie_ids_list, index=self.movie_ids_list)           


    def make_recommendations(self, _id):
        '''
        INPUT:
        _id - either a user or movie id (int), depends on method

        OUTPUT:
        rec_ids - (array) a list or numpy array of recommended movies like the
                       given movie, or recs for a user_id given
        rec_names - (list) a list of recommended movie names              
        '''
        print()
        print('Making recommendation .......')
        print('Method: {} id_type: {}'.format(self.method, self._idtype))
        print() 

        if(self._idtype == 'user'):
            if(self.method=='rank-base'):
                
                self.rec_ids = rf.get_top_article_ids(_id, n=self.rec_num, data=self.ranked_df)
            # create some surprise
                #self.rec_ids = np.random.choice(rec_ids_tmp, self.rec_num, replace=False)
                self.rec_names = rf.get_article_names(self.rec_ids, df=self.df)

            if(self.method=='funk-SVD'):

                if(_id in self.user_ids_series):
                    #funk svd recomm
                    user_unseen_movie_id = rf.svd_recommendation(_id, self.train_data_df, self.user_mat, self.movie_mat, self.rec_num)
                    self.rec_ids = np.random.choice(user_unseen_movie_id, self.rec_num, replace=False)
                    self.rec_names = rf.get_movie_names(self.rec_ids, self.movies)
                else:
                    print('user_id: {} not found, try another one'.format(_id))
                    return None                


if __name__ == '__main__':
    # test different parts to make sure it works

    df, df_content = rf.load_data('data/user-item-interactions.csv', 'data/articles_community.csv')

    # Method user-base
    print('Method mix-------------------------------------------------')
    rec = Recommender(method='user-base')
    #print('here')
    rec.fit(movies=movies_, reviews=reviews_)
    print(rec.make_recommendations(8))  # in data
    print(rec.make_recommendations(1))  # not in data