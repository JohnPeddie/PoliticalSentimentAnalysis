import requests
import config
import pandas as pd


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


def fetchSubreddit(subreddit, headers):
    res = requests.get("https://oauth.reddit.com/r/" + subreddit + "/hot",
                       headers=headers)
    return res

def buildDataFrame(res):
    df = pd.DataFrame()
    for post in res.json()['data']['children']:
        df = df.append({
            'subreddit': post['data']['subreddit'],
            'title': post['data']['title'],
            'selftext': post['data']['selftext']
        }, ignore_index=True)
    return df



def main():
    dfPython = buildDataFrame(fetchSubreddit("python", buildAPIConnection()))
    print(dfPython)


if __name__ == "__main__":
    main()
