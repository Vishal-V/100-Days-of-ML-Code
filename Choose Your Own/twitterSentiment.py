from textblob import TextBlob
import tweepy
import csv

consumer_api = '' #Consumer API
consumer_api_secret = '' #Consumer API secret

access_token = '' #Access token
access_token_secret = '' #Access token secret

auth = tweepy.OAuthHandler(consumer_api, consumer_api_secret)
auth.set_access_token(access_token,access_token_secret)

api = tweepy.API(auth)

tweets = api.search('happy')

with open('tweet_this.csv', 'w') as tweet_file:
	twitterit = csv.writer(tweet_file)
	for tweet in tweets:
		twitterit.writerow(tweet.text.encode('utf-8'))
		wiki = TextBlob(tweet.text)
		print(wiki.sentiment)