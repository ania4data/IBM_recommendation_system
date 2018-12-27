import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import project_tests as t
import pickle
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
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
    def __init__(self, method='user-base', _idtype='user', rec_num=10, word_filter='visualization'):
        '''
        This function initialize the class
        INPUT:
        method - (str) 'user-base': user similarity, 'content-base': item NLP content, 'rank-base': most popularity, 'rank-base-content': rank-base with filtered NLP
        _idtype - (str) 'user' or 'item'
                        'user' only compatible with 'user-base'/'rank-base'/'rank-base-content' methods
                        'item' only compatible with 'content-base' methods
        rec_num - (int) number of recommendations for output
        '''
        self.method = method
        self._idtype = _idtype
        self.rec_num = rec_num
        self.word_filter = word_filter

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
        self.df_content_clean = self.df_content.drop_duplicates(subset=['article_id','doc_full_name', 'doc_status'])

        print('Fitting .......')
        print('Method: {} id_type: {}'.format(self.method, self._idtype))
        print()


        if(self._idtype == 'user'):
            if((self.method == 'rank-base') or (self.method == 'rank-base-content')):
                self.ranked_df = rf.get_rank_data(df=self.df)
            if((self.method == 'rank-base-content')):
                self.content_tokens_dict = rf.get_token_df_content(self.df_content_clean)
                self.len_token = rf.token_length(self.df_content_clean, self.content_tokens_dict)  ##??
                self.article_bag_of_words_vec, self.global_words_update, self.content_tokens_dict_update_1 = rf.get_bag_words_vec(self.df_content_clean, self.content_tokens_dict)

            if((self.method == 'user-base')):
                self.user_item, self.user_item_matrix = rf.create_user_item_matrix(self.df)

        if(self._idtype == 'item'):
            if(self.method == 'content-base'):
                self.content_tokens_dict = rf.get_token_df_content(self.df_content_clean)
                self.article_bag_of_words_vec, self.global_words_update, self.content_tokens_dict_update_1 = rf.get_bag_words_vec(self.df_content_clean, self.content_tokens_dict)



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
                
                self.rec_ids = rf.get_top_article_ids(_id, self.rec_num, self.ranked_df)
                self.rec_names = rf.get_article_names(self.rec_ids, self.df)
                return self.rec_ids, self.rec_names

            if(self.method=='rank-base-content'):
                self.top_df_filtered = rf.rec_ranked_word_specific(self.global_words_update, self.content_tokens_dict_update_1, self.df, self.rec_num, self.word_filter)
                return self.top_df_filtered

            if(self.method=='user-base'):

                self.most_similar_users, self.similarity_df = rf.find_similar_users(_id, self.user_item) 
                self.rec_ids, self.rec_names = rf.user_user_recs_part2(_id, self.user_item, self.df, self.rec_num)
                return self.rec_ids, self.rec_names


        if(self._idtype == 'item'):
            if(self.method=='content-base'):
                self.rec_ids, self.rec_names = rf.make_content_recs(_id, self.article_bag_of_words_vec, self.df, self.df_content_clean, self.rec_num)
                return self.rec_ids, self.rec_names

if __name__ == '__main__':

    df, df_content = rf.load_data('data/user-item-interactions.csv', 'data/articles_community.csv')
    # print(df.head(2))
    # print(df_content.head(2))

    print('Method user-base-------------------------------------------------')
    rec = Recommender(method='user-base')
    rec.fit(df_content, df)
    ids, names = rec.make_recommendations(1)
    print(ids, names)  

    print('Method rank-base-------------------------------------------------')
    rec = Recommender(method='rank-base', rec_num=5)
    rec.fit(df_content, df)
    ids, names = rec.make_recommendations(1)
    print(ids, names)  

    print('Method rank-content-base-------------------------------------------------')
    rec = Recommender(method='rank-base-content', rec_num=10, word_filter='flow')
    rec.fit(df_content, df)
    print(rec.make_recommendations(10))  

    print('Method content-base-------------------------------------------------')
    rec = Recommender(method='content-base', _idtype='item', rec_num=10)
    rec.fit(df_content, df)
    print(rec.make_recommendations(100))  