# Political (sentiment) Analysis (of) Tweets - (PAT)
A python ML and NLP project that attempts to determine political sentiment/allignment from a message trained on reddit posts and tweets.
Current demo link: https://trumpspoliticalcompassdemo.netlify.app/ and: https://fivepoliticalleadersonthecompass.netlify.app/

## How to - Initial Set Up:
1. Clone the repo,
2. Set up a reddit developer account and create a new project to be used to access the reddit API: https://towardsdatascience.com/how-to-use-the-reddit-api-in-python-5e05ddfd1e5c
3. Create a file called "config.py" in the main directory and include the following infomation:
```
redditUsername = "your reddit username"
redditPassword = "your reddit password"
redditAPIClientID = "your reddit api client ID"
redditAPIClientSecret = "your api client secret"
```
4. Set up a virtual environment: python3 -m venv env
5. Install the requirements.txt: pip install -r requirements.txt

## How to - Running the File:
1. Gathering Democrats/Republicans training data: Run main.py (subreddits can be changed as long as one is left leaning and the other is right leaning)
2. Gathering Authoritarian/Libertarian training data: Run mainAuthLib.py (again, subreddits can be changed as long is top leaning and the other is bottom)
3. These two files will generate CSVs of training data,
4. Run PAT.py to train the AI and generate the political compasses, play around with various functions in this file to get the result you desire.
5. Final compass is generated as index.html, display this in browser.

## ETC
This is very work in progress and my first attempt at a proper ML project, this is far from a final state and a lot more work is required for this to be representative, accurate and efficient. Please understand this and expect further changes.
