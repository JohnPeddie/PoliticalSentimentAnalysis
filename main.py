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
    repScores = []
    demScores = []
    for repScore in (df.loc[df['subreddit'] == 'republicans', 'flesch_reading_ease']):
        if repScore > -100:
            repScores.append(repScore)
    for demScore in (df.loc[df['subreddit'] == 'democrats', 'flesch_reading_ease']):
        if demScore > -100:
            demScores.append(demScore)

    print("Dem score mean: ", np.mean(demScores))
    print("Rep score mean: ", np.mean(repScores))
    plt.plot(repScores,"r-")
    plt.plot(demScores,"b-")
    plt.show()

def frequencies(df,subreddit):
    additionalPoliticsEnglishStop = ['www', 'things', 'does', 'x200b', 'amp', 'want', 'watch',
                                        'just', 'like', 'https', 'com', 'republican', 'republicans',
                                        'libertarians', 'democrats', 'democrat', 'people', 'libertarian',
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
    url_democrats = "https://oauth.reddit.com/r/Democrats/.json"
    url_republicans = "https://oauth.reddit.com/r/Republicans/.json"
    print("Fetching Republican Posts ...")
    get_republicans = get_subreddit(url_republicans, 50, buildAPIConnection())
    print("Fetching Democratic Posts ...")
    get_democrats = get_subreddit(url_democrats, 50, buildAPIConnection())
    repDF = buildDataFrame(get_republicans)
    demDF = buildDataFrame(get_democrats)

    masterDF = fetchReadingGrade(fetchReadibilityScore(pd.merge(repDF, demDF,'outer')))
    graphStats(masterDF)
    print(frequencies(masterDF,'democrats'))
    demsWordcount = pd.DataFrame(frequencies(demDF,'democrats'), columns=["Frequency"])
    repsWordcount = pd.DataFrame(frequencies(repDF,'republicans'), columns=["Frequency"])

    demsTop15 = demsWordcount.head(15)
    demsTop15.sort_values('Frequency', ascending=True).plot(kind="barh")
    plt.title("Top 15 Words in Democrats subreddit")
    plt.xlabel("Frequency (Count)")
    plt.ylabel("Word")
    plt.show()

    repsTop15 = repsWordcount.head(15)
    repsTop15.sort_values('Frequency', ascending=True).plot(kind="barh")
    plt.title("Top 15 Words in Republicans subreddit")
    plt.xlabel("Frequency (Count)")
    plt.ylabel("Word")
    plt.show()

    print(masterDF)


if __name__ == "__main__":
    main()
