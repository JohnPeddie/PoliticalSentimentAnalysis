#Same as main.py but a test to see if I can do the same thing with Auth and Lib views

import requests
import sklearn.feature_extraction.text

import config
import pandas as pd
import textstat
import time
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.feature_extraction import text


def buildAPIConnection():
    auth = requests.auth.HTTPBasicAuth(config.redditAPIClientID, config.redditAPIClientSecret)
    data = {'grant_type': 'password',
            'username': config.redditUsername,
            'password': config.redditPassword}
    headers = {'User-Agent': 'PythonPostPuller/0.0.1'}
    res = requests.post('https://www.reddit.com/api/v1/access_token',
                        auth=auth, data=data, headers=headers)
    TOKEN = res.json()['access_token']
    headers = {**headers, **{'Authorization': f"bearer {TOKEN}"}}
    return headers


def get_subreddit(url, n_pulls, headers):
    # Create empty templates
    posts = []
    after = None
    duplicates = []
    # Create a loop that does max 25 requests per pull
    for pull_num in range(n_pulls):
        print("Pulling data attempted", pull_num + 1, "time(s)")

        if after == None:
            new_url = url  # base case
        else:
            new_url = url + "?after=" + after  # subsequent iterations

        res = requests.get(new_url, headers=headers)

        if res.status_code == 200:
            subreddit_json = res.json()  # Pull JSON
            posts.extend(subreddit_json['data']['children'])  # Get subreddit posts
            after = subreddit_json['data']['after']  # 'after' = ID of the last post in this iteration
        else:
            print("We've run into an error. The status code is:", res.status_code)
            break

        time.sleep(1)

    return (posts)


def buildDataFrame(posts):
    df = pd.DataFrame()
    for post in posts:
        fullPostTitle = re.sub(r"\r\n", " ", post['data']['title'])
        fullPostTitle = re.sub("[^0-9A-Za-z ]","",fullPostTitle)

        if (type(post['data']['selftext']) == str) & ("poll" not in post['data']['selftext']) & ("http" not in post['data']['selftext']):
            fullPostTitle = re.sub(r"\r\n", " ",post['data']['title'] + ' ' + post['data']['selftext'])
            fullPostTitle = re.sub("[^0-9A-Za-z ]","",fullPostTitle)

        df = df.append({
            'subreddit': post['data']['subreddit'],
            'title': fullPostTitle,
            'selftext': post['data']['selftext'],
            'link_flair_text': post['data']['link_flair_text']
        }, ignore_index=True)
    return df


def cleanTextFields(df):
    items = []
    for item in df["title"]:
        items.append(item)
    print(items)


def fetchReadibilityScore(df):
    textFleschReadingEase = []
    for item in df["title"]:
        textFleschReadingEase.append(textstat.flesch_reading_ease(item))
    df["flesch_reading_ease"] = textFleschReadingEase
    return df

def fetchReadingGrade(df):
    textFleschKincaidGrade =[]
    for item in df["title"]:
        textFleschKincaidGrade.append(textstat.flesch_kincaid_grade(item))
    df["flesch_kincaide_grade"] = textFleschKincaidGrade
    return df

def graphStats(df):
    libscores = []
    authscores = []
    for libscore in (df.loc[df['subreddit'] == 'Libertarian', 'flesch_reading_ease']):
        if libscore > -100:
            libscores.append(libscore)
    for authscore in (df.loc[df['subreddit'] == 'AuthoritariansDiscuss', 'flesch_reading_ease']):
        if authscore > -100:
            authscores.append(authscore)

    print("auth score mean: ", np.mean(authscores))
    print("lib score mean: ", np.mean(libscores))
    plt.plot(libscores,"r-")
    plt.plot(authscores,"b-")
    plt.show()

def frequencies(df,subreddit):
    additionalPoliticsEnglishStop = ['www', 'things', 'does', 'x200b', 'amp', 'want', 'watch',
                                        'just', 'like', 'https', 'com', 'libertarian', 'libertarians',
                                        'libertarians', 'authoritarians', 'authoritarian', 'people', 'libertarian',
                                        'says', 'say', 'did', 'this', 'conservative', 'conservatives']

    englishPoliticsStopList = text.ENGLISH_STOP_WORDS.union(additionalPoliticsEnglishStop)
    cv = CountVectorizer(stop_words=englishPoliticsStopList)
    cv_fit = cv.fit_transform(df.loc[df['subreddit'] == subreddit, 'title'])
    # Converts text to array form
    cvText = pd.DataFrame(cv_fit.toarray(), columns= cv.get_feature_names())

    # See word counts
    word_counts = cvText.sum().sort_values(0, ascending=False)

    return word_counts



def main():
    url_auth = "https://oauth.reddit.com/r/AuthoritariansDiscuss/.json"
    url_lib = "https://oauth.reddit.com/r/Libertarian/.json"
    print("Fetching libertarian Posts ...")
    get_libertarians = get_subreddit(url_lib, 50, buildAPIConnection())
    print("Fetching AuthoritariansDiscuss Posts ...")
    get_authoritarians = get_subreddit(url_auth, 50, buildAPIConnection())
    libDF = buildDataFrame(get_libertarians).drop_duplicates()
    authDF = buildDataFrame(get_authoritarians).drop_duplicates()

    masterDF = fetchReadingGrade(fetchReadibilityScore(pd.merge(libDF, authDF,'outer')))
    graphStats(masterDF)
    print(frequencies(masterDF,'AuthoritariansDiscuss'))
    authsWordcount = pd.DataFrame(frequencies(authDF,'AuthoritariansDiscuss'), columns=["Frequency"])
    libsWordcount = pd.DataFrame(frequencies(libDF,'Libertarian'), columns=["Frequency"])

    authsTop15 = authsWordcount.head(15)
    authsTop15.sort_values('Frequency', ascending=True).plot(kind="barh")
    plt.title("Top 15 Words in authoritarians subreddit")
    plt.xlabel("Frequency (Count)")
    plt.ylabel("Word")
    plt.show()

    libsTop15 = libsWordcount.head(15)
    libsTop15.sort_values('Frequency', ascending=True).plot(kind="barh")
    plt.title("Top 15 Words in libertarians subreddit")
    plt.xlabel("Frequency (Count)")
    plt.ylabel("Word")
    plt.show()

    print(masterDF)
    masterDF.to_csv('./data/masterDFAuthLib.csv', index=False)


if __name__ == "__main__":
    main()
