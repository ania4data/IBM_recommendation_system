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

def load_data(user_article_pth, article_pth):

	df = pd.read_csv(user_article_pth)
	#'data/user-item-interactions.csv'
	df_content = pd.read_csv(article_pth)
	#'data/articles_community.csv'
	del df['Unnamed: 0']
	del df_content['Unnamed: 0']

	email_encoded = email_mapper(data = df)
	del df['email']
	df['user_id'] = email_encoded
	df.article_id = df.article_id.astype('int64')

	return df, df_content

def email_mapper(data = df):

    coded_dict = dict()
    cter = 1
    email_encoded = []
    
    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter+=1
        
        email_encoded.append(coded_dict[val])
    return email_encoded

def get_top_articles(n, df=df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles 
    
    '''
    top_ids_list = get_top_article_ids(n, df)
    top_n_df = df.groupby(by=['article_id']).count().sort_values(by='title', ascending=False).reset_index().iloc[0:n]
    top_n_df = top_n_df.reset_index() 
    unique_id_title = df[['article_id','title']].drop_duplicates()
    unique_id_title = unique_id_title[unique_id_title.article_id.isin(top_ids_list)]
    top_articles = list(top_n_df.merge(unique_id_title, on='article_id', suffixes=('_', '')).title.values)

    return top_articles # Return the top article titles from df (not df_content)

def get_top_article_ids(n, df=df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article ids
    
    '''
    df.article_id = df.article_id.astype('int64')
    ids_list = list(df.groupby(by=['article_id']).count().sort_values(by='title', ascending=False)
                    .reset_index().iloc[0:n].article_id.values)
    top_articles = ids_list
    
    return top_articles # Return the top article ids   


def create_user_item_matrix(df):
    '''
    INPUT:
    df - pandas dataframe with article_id, title, user_id columns
    
    OUTPUT:
    user_item - user item dataframe
    user_item_matrix - user item matrix
    Description:
    
    Return a matrix and a dataframe with user ids as rows and article ids on the columns with 1 values where a user interacted with 
    an article and a 0 otherwise
    '''
    df_no_duplicate = df.drop_duplicates()
    user_id_list = df_no_duplicate.user_id.unique()  #unique
    article_id_list = df_no_duplicate.article_id.unique() #unique
    
    user_item_matrix = np.zeros((len(user_id_list), len(article_id_list)))
    user_item = pd.DataFrame(user_item_matrix, columns=article_id_list, index=user_id_list)
    for user_id, article_id in df_no_duplicate[['user_id', 'article_id']].values:
        row = np.where(user_id_list == user_id)[0][0]
        col = np.where(article_id_list == article_id)[0][0]
        user_item_matrix[row, col] = 1
        user_item.loc[user_id, article_id] = 1
    
    return user_item, user_item_matrix # return the user_item matrix 


def find_similar_users(user_id, user_item=user_item):
    '''
    INPUT:
    user_id - (int) a user_id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    most_similar_users - (list) an ordered list where the closest users (largest dot product users)
                    are listed first
    
    similarity_df - a dataframe containing each user and its neighbors order by similarity

    Description:
    Computes the similarity of every pair of users based on the dot product
    Returns an ordered
    '''
    user1 = user_id
    user_list = user_item.index
    user_similarity_array = []
    for user2 in user_list:  # removing owns or duplicates
        similarity = np.dot(user_item.loc[user1], user_item.loc[user2])
        user_similarity_array.append([user1, user2, similarity])
        
    similarity_df = pd.DataFrame(user_similarity_array, columns=['user1', 'user2', 'similarity'])  
    similarity_df = similarity_df.sort_values(by='similarity', ascending=False)
    similarity_df = similarity_df[similarity_df.user2 != user1] # remove user1==user2
    most_similar_users = similarity_df.user2.values
    
    return most_similar_users, similarity_df  # return a list of the users in order from most to least similar


def get_article_names(article_ids, df=df):
    '''
    INPUT:
    article_ids - (list) a list of article ids
    df - (pandas dataframe) df as defined at the top of the notebook
    
    OUTPUT:
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the title column)
    '''
    article_names = []
    df_clean_id_title= df[['article_id','title']].drop_duplicates()
    article_name_df = df_clean_id_title[df_clean_id_title.article_id.isin(article_ids)]
    
    for article_id in article_ids:
        article_names.append(article_name_df[article_name_df['article_id']==article_id].title.values[0])

    return article_names # Return the article names associated with list of article ids

def get_user_articles(user_id, user_item=user_item):
    '''
    INPUT:
    user_id - (int) a user id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    article_ids - (list) a list of the article ids seen by the user
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the doc_full_name column in df_content)
    
    Description:
    Provides a list of the article_ids and article titles that have been seen by a user
    '''
    df_user = user_item.loc[user_id]
    article_ids = list(df_user[df_user!=0.0].index)
    article_names = get_article_names(article_ids, df=df)
    
    return article_ids, article_names # return the ids and names

def user_user_recs(user_id, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user
    
    OUTPUT:
    recs - (list) a list of recommendations for the user
    
    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found
    
    Notes:
    Users who are the same closeness are chosen arbitrarily as the 'next' user
    
    For the user where the number of recommended articles starts below m 
    and ends exceeding m, the last items are chosen arbitrarily
    
    '''
    recs = []
    all_articles = list(user_item.columns)
    user1 = user_id
    article_ids_user1_seen, _ = get_user_articles(user1, user_item=user_item)
    
    most_similar_users, similarity_df = find_similar_users(user1, user_item=user_item)
    x = similarity_df.sort_values(by='similarity', ascending=False)
    # get unique, then go through uniques if multiple and pick random from them
    unique_sim = x.similarity.unique()   
    
    # loop through all unique similarities
    for unique in unique_sim:
        y = similarity_df[similarity_df.similarity==unique]
        user2_list = y.user2.values
        unique_length= y.shape[0]
        # arbitary pick next user, with same similarities
        random_pick_similar = np.random.choice(user2_list, unique_length, replace=False)
        for user2 in random_pick_similar:
            article_ids_user2_seen, _ = get_user_articles(user2, user_item=user_item)
            article_ids_user1_notseen = np.setdiff1d(article_ids_user2_seen, article_ids_user1_seen)
            start_length = len(recs)
            # to avoid similar pointer,recs_tmp should not point same place as recs, so need only copy
            recs_tmp = recs.copy()   
            recs_tmp.extend(article_ids_user1_notseen)
            recs_tmp = list(set(recs_tmp))
            end_length = len(recs_tmp)

            # check before and after length, if more than m random pick
            if(end_length>=m):
                extra_count_needed = m - start_length
                list_get_rest = np.random.choice(article_ids_user1_notseen, extra_count_needed, replace=False)
                recs.extend(list_get_rest)
                recs = list(set(recs))
                return recs

            if(end_length<m):
                recs.extend(article_ids_user1_notseen)
                recs = list(set(recs))
                
    # user never meet required m rec
    return recs
   
def get_top_sorted_users(user_id, df=df, user_item=user_item):
    '''
    INPUT:
    user_id - (int)
    df - (pandas dataframe) df as defined at the top of the notebook 
    user_item - (pandas dataframe) matrix of users by articles: 
            1's when a user has interacted with an article, 0 otherwise
    
            
    OUTPUT:
    neighbors_df - (pandas dataframe) a dataframe with:
                    user2 - is a neighbor user_id
                    similarity - measure of the similarity of each user to the provided user_id
                    unique_total - the number of unique articles viewed by the user 
                    overall_total - the number of overall articles viewed by the user 
                    
    Other Details - sort the neighbors_df by the similarity and then by number of interactions where 
                    highest of each is higher in the dataframe
     
    '''
    #unique interaction count
    x = df.drop_duplicates().groupby(by='user_id').count()  
    unique_total = x.reset_index()[['user_id','article_id']].rename(columns={"article_id": "unique_total"})
    #overall interaction count (even if user visited same unique article multiple time)
    x = df.groupby(by='user_id').count() 
    overall_total = x.reset_index()[['user_id','article_id']].rename(columns={"article_id": "overall_total"})
    merged_counts = unique_total.merge(overall_total, on='user_id')
    _, similarity_df = find_similar_users(user_id, user_item=user_item)
    z = similarity_df.merge(merged_counts, left_on='user2',right_on='user_id').drop(columns=['user_id'])
    # put priority by overall total (test staisfaction, otherwise think unique total count is more important)
    z = z.sort_values(by=['similarity','overall_total','unique_total'],ascending=False) 
    neighbors_df = z[z.user1==user_id]
    
    return neighbors_df # Return the dataframe specified in the doc_string

def user_user_recs_part2(user_id, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user
    
    OUTPUT:
    recs - (list) a list of recommendations for the user by article id
    rec_names - (list) a list of recommendations for the user by article title
    
    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found
    
    Notes:
    * Choose the users that have the most total article interactions 
    before choosing those with fewer article interactions.

    * Choose articles with the articles with the most total interactions 
    before choosing those with fewer total interactions. 
   
    '''
    recs = []
    all_articles = list(user_item.columns)
    user1 = user_id
    article_ids_user1_seen, _ = get_user_articles(user1, user_item=user_item)
    
    neighbors_df = get_top_sorted_users(user1, df=df, user_item=user_item)
    user2_list = neighbors_df.user2.values

    for user2 in user2_list:
        article_ids_user2_seen, _ = get_user_articles(user2, user_item=user_item)
        article_ids_user1_notseen = np.setdiff1d(article_ids_user2_seen, article_ids_user1_seen)
        start_length = len(recs)
        # to avoid similar pointer,recs_tmp should not point same place as recs, so need only copy
        recs_tmp = recs.copy()   
        recs_tmp.extend(article_ids_user1_notseen)
        recs_tmp = list(set(recs_tmp))
        end_length = len(recs_tmp)

        if(end_length>=m):
            extra_count_needed = m - start_length
            top_100_articles = get_top_article_ids(100, df=df)
            list1_df = pd.DataFrame(np.array(top_100_articles))
            list2_df = pd.DataFrame(np.array(article_ids_user1_notseen))
            # intersection of the two lists but with top_100_priority and get extra_count_needed from top
            list_get_rest = list1_df.merge(list2_df)[0].values[0:extra_count_needed]            
            recs.extend(list_get_rest)
            recs = list(set(recs))
            rec_names = get_article_names(recs)
            return recs, rec_names

        if(end_length<m):
            recs.extend(article_ids_user1_notseen)
            recs = list(set(recs))

    #if user never meet m recommendation
    rec_names = get_article_names(recs)
    return recs, rec_names

def get_token_df_content(df_content_clean_copy):
    '''
    INPUT:
    df_content_clean_copy - a dataframe of article description,
    with unique article id
        
    OUTPUT:
    content_tokens_dict - (dictionary) a dictionary that connect the article_id 
    to tokens
       
    Description:
    Function creates a dictionary that connect the article_id to 
    tokens from doc_full_name + doc_description + doc_status
    a list of word tokens that are lemmatized and cleaned from stopwords
    '''
    df_less_col = df_content_clean_copy[['article_id', 'doc_status', 'doc_full_name', 'doc_description']].copy()
    content_tokens_dict = {}
    list_of_articles = df_content_clean_copy.article_id.values
    for article in list_of_articles:
        article_df = df_less_col[df_less_col.article_id == article]               
        try:
            try:
                article_text = article_df.doc_full_name.values[0] + ' ' + article_df.doc_description.values[0]
                content_tokens_dict[article] = tokenizer.tokenize(article_text)
            except:
                article_text = article_df.doc_full_name.values[0]
                content_tokens_dict[article] = tokenizer.tokenize(article_text)
        except:
            break
    return content_tokens_dict

def token_length(df_content_clean_copy, dict_ = content_tokens_dict):
    '''
    INPUT:
    df_content_clean_copy - a dataframe of article description,
    with unique article id
    
    content_tokens_dict - (dictionary) a dictionary that connect the article_id 
    to tokens 
    
    OUTPUT:
    len_token - (list) a list of token lengths associate with article_ids 
    to tokens
    '''    
    len_token = []
    for article in df_content_clean_copy.article_id.values:
        len_token.append(len(dict_[article]))
        if(len(dict_[article])==1):
            print(dict_[article])
    return len_token

def get_bag_words_vec(df_content_clean_copy)

	global_words = []
	for article in df_content_clean_copy.article_id.values:
	    global_words.extend(content_tokens_dict[article])
	global_words_counter = Counter(global_words)

	count = 0
	content_tokens_dict_update_1 = {}
	for article in df_content_clean_copy.article_id.values:
	    tokens = content_tokens_dict[article]
	    tokens_update_1 = []
	    for token in tokens:
	        if(global_words_counter[token]>1):
	            tokens_update_1.append(token)
	    content_tokens_dict_update_1[article] =  tokens_update_1  

	global_words_update = []
	for article in df_content_clean_copy.article_id.values:
	    global_words_update.extend(content_tokens_dict_update_1[article])

	article_bag_of_words_vec = {}
	for article in df_content_clean_copy.article_id.values:
	    bag_of_words_coded = np.zeros(len(global_words_update))
	    tokens = content_tokens_dict_update_1[article]
	    for token in tokens:
	        idx = global_words_update.index(token)
	        bag_of_words_coded[idx] = 1.0
	    article_bag_of_words_vec[article] = bag_of_words_coded  

	return article_bag_of_words_vec, global_words_update


def rec_ranked_word_specific(global_words_update, content_tokens_dict_update_1, df=df, word_filter='visualization', n_rec=10):
    '''
    INPUT:
    global_words_update - (list) a list of global unique tokens
    content_tokens_dict_update_1 - a dictionary for each article_id to a bag of words
    df - dataframe of user/article viewing data
    word_filter - (str)  an input keyword for filtering
    n_rec - (int) number of recommendation
    
    OUTPUT:
    top_df_filtered - a dataframe containing each article filtered by the word_filter sorted by
    total viewing count

    Description:
    Get a word, pass a through tokenizer and find articles that have the word mentioned in them
    pick those article and get top n_rec of them sorted by viewing count, makes sure article ids 
    from df_content also exist in df
    '''    
    
    overall_total = df.groupby(by='article_id').count().sort_values(by=['user_id'], ascending=False).reset_index()\
                    .rename(columns={"title": "total_view_count"})   
    article_ids_filtered =[]
    article_ids = list(content_tokens_dict_update_1.keys())
    try:
        word = tokenizer.tokenize(word_filter)[0]
        print('------------', word)
        if(word in global_words_update):     
            for article in article_ids:
                if(word in content_tokens_dict_update_1[article]):
                    article_ids_filtered.append([word, article])
            article_found_df = pd.DataFrame(article_ids_filtered, columns=['word','article_id_found'])
            z = article_found_df.merge(overall_total, left_on='article_id_found',right_on='article_id').drop(columns=['user_id', 'article_id'])
            merged_article_found_df_count = z.sort_values(by=['total_view_count'], ascending=False)
            article_ids_output = merged_article_found_df_count.article_id_found.values
            article_view_count = merged_article_found_df_count.total_view_count.values
            print('top {} viewed articles with word {}, \n'.format(n_rec, word_filter))
            print('------------')
            print('Article ids:\n', article_ids_output)
            print('------------')
            print('Article names:\n', get_article_names(article_ids_output))
            print('------------')
            print('Article total view count: \n', article_view_count)
            top_df_filtered = merged_article_found_df_count[0: n_rec]
            return  top_df_filtered
       
        else:
            print('Try another word')    
    except:
        print('Try another word')
        

def find_similar_articles(article_id, bag_words_dict_vec=article_bag_of_words_vec, df=df):
    '''
    INPUT:
    article_id - (int) a article_id from df_content
    ag_words_dict_vec - a dictionary for each article_id to a bag of words
    df - dataframe of user/article viewing data
    
    OUTPUT:
    similarity_df - a dataframe containing each article and its neighbors order by similarity, 
    and total viewing count

    Description:
    Computes the similarity of every pair of articles based on the dot product/ and viewing count
    '''
    article1 = article_id
    n = df.article_id.nunique()
    article_list = list(bag_words_dict_vec.keys())
    article_similarity_array = []
    overall_total = df.groupby(by='article_id').count().sort_values(by=['user_id'], ascending=False).reset_index()\
                    .rename(columns={"title": "total_view_count"})   
    try:
        test = bag_words_dict_vec[article1]
        for article2 in article_list:  # removing owns or duplicates
            similarity = np.dot(bag_words_dict_vec[article1], bag_words_dict_vec[article2])
            article_similarity_array.append([article1, article2, similarity])

        similarity_df = pd.DataFrame(article_similarity_array, columns=['article1', 'article2', 'similarity'])  
        similarity_df = similarity_df[similarity_df.article2 != article1] # remove article1==article2
        z = similarity_df.merge(overall_total, left_on='article2',right_on='article_id').drop(columns=['user_id', 'article_id'])
        neighbors_df = z[z['article1']==article1] 
        neighbors_df = neighbors_df.sort_values(by=['similarity', 'total_view_count'], ascending=False)        
        return neighbors_df
    
    except:
        print('try an article id belong to the df_content (article info file)')
        return None


def make_content_recs(user_id, article_id, bag_words_dict_vec=article_bag_of_words_vec, df=df, n_rec=10):
    '''
    INPUT:
    user_id - (int)  user_id 
    article_id - (int) a article_id
    ag_words_dict_vec - a dictionary for each article_id to a bag of words
    df - dataframe of user/article viewing data
    n_rec - (int) number of recommendations
    
    OUTPUT:
    rec_ids - (list) a list of recommendation id
    rec_names - (list) a list of recommnedation name

    Description:
    report the top n_rec recommendations similar to article_id sorted by similarity in content
    and then total viewing count
    Note:
    user_id has no active impact on suggestion, this role is given to article_id
    '''
    try:
        
        rec_ids = find_similar_articles(article_id, bag_words_dict_vec=article_bag_of_words_vec, df=df)[0: n_rec]\
                  .article2.values
        rec_names = get_article_names(rec_ids)
        return rec_ids, rec_names
    except:
        return None