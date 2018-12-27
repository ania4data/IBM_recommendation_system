# Recommendation system for IBM Watson Studio

<p align="center"> 
<img src="https://github.com/ania4data/Recommendation_systems/blob/master/IBM_recommendation/images/ibm-watson.jpg", style="width:30%">
</p>      

In this project I have introduced multiple recommendation systems, including Rank-base (item popularity), User-base (user similarity), Content-base (item similarity, NLP), and SVD (matrix factorization) to study the will be looking at the interactions that users have with articles on the IBM Watson Studio platform. Below you can see an example of what the dashboard could look like displaying articles on the IBM Platform.

<p align="center"> 
<img src="https://github.com/ania4data/Recommendation_systems/blob/master/IBM_recommendation/images/dashboard.png", style="width:30%">
</p>

# Repository layout

```
│   LICENSE
│   project_tests.py
│   README.md
│   Recommendations_with_IBM.html
│   Recommendations_with_IBM.ipynb
│   recommender.py
│   recommender_functions_utils.py
│   top_10.p
│   top_20.p
│   top_5.p
│   user_item_matrix.p
│
├───backup
│       Recommendations_with_IBM_122318.ipynb
├───data
│       articles_community.csv
│       user-item-interactions.csv
│
├───images
│       dashboard.png
│       ibm-watson.jpg
│
└───utils
        tokenizer.py

```
## General repository content:

- data folder: two csv files `user-item-interactions.csv` for user/article interaction and `articles_community.csv` which contains information with regard to articles.
- images folder: contains all static images for README.md file
- utils folder: contains the `tokenizer.py` file for preprocessing/tokenizing input text
- LICENSE file
- README.md file
- test files: `top_*.p files`/`user_item_matrix.p` and `project_tests.py` for tests running through the notebook
- Jupter notebook: `Recommendations_with_IBM.ipynb` contains recommendations notebook
- `recommender.py`: is the conversion of the notebook file into the class 
    - for Users: user-base (user similarity), rank-base (popularity), rank-content-base (popularity/keyword/NLP)
    - for Items: content-base(article similarity/NLP)
- `recommender_functions_utils.py`: all necessary functions for `recommender.py` gathered here

## How to run:

- clone the repository, `git clone https://github.com/ania4data/Recommendation_systems.git`
- within IBM_recommendation directory, run `python recommender.py`