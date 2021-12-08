# using pushshift API to gather data from banned subbreddits


from psaw import PushshiftAPI
import datetime
import pandas as pd
import re


def fetchData(postedBefore, postedAfter, subreddit, limit, api):
    query = api.search_submissions(subreddit=subreddit, after=postedAfter, before=postedBefore, limit=limit)
    submissions = list()
    for element in query:
        element.d_['subreddit'] = subreddit
        submissions.append(element.d_)
    return submissions

def buildDataFrame(posts):
    df = pd.DataFrame()
    for post in posts:

        fullPostTitle = re.sub(r"\r\n", " ", post['title'])
        fullPostTitle = re.sub("[^0-9A-Za-z ]","",fullPostTitle)

        if (type(post['selftext']) == str) & ("poll" not in post['selftext']) & ("http" not in post['selftext']):
            fullPostTitle = re.sub(r"\r\n", " ",post['title'] + ' ' + post['selftext'])
            fullPostTitle = re.sub("[^0-9A-Za-z ]","",fullPostTitle)

        df = df.append({
            'subreddit': post['subreddit'],
            'title': fullPostTitle,
            'selftext': post['selftext'],

        }, ignore_index=True)
    return df

def mergeCSVs(masterDF,CSV):
    toMerge = pd.read_csv(CSV)
    merged = masterDF.append(toMerge)
    return merged


def main():
    api = PushshiftAPI()
    postedAfter = int(datetime.datetime(2019, 6, 1).timestamp())
    postedBefore = int(datetime.datetime(2021, 1, 30).timestamp())
    subreddit = "Authoritarianism"
    limit = 1000
    posts = fetchData(postedBefore, postedAfter, subreddit, limit, api)
    masterDF = buildDataFrame(posts)
    masterAuthLibDF = mergeCSVs(masterDF,'data/masterDFAuthLib.csv')
    masterAuthLibDF.to_csv('./data/masterDFAuthLib.csv', index=False)



if __name__ == "__main__":
    main()
