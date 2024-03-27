
from mpi4py import MPI
from collections import defaultdict
import argparse
import os
import time
def maxFinder(dic):
    if not dic:
        return None, None
    else:
        max_key, max_value = max(dic.items(), key=lambda x: x[1])
        return max_key, max_value

def splitTweet(line):
    date = None
    hour = None
    sentiment = None
    if "created_at" in line:
            date, hour = line.split('"created_at":"')[1].split('"',1)[0].split(":")[0].split("T")
            hour = hour[:2]
    if '"score"' not in line:
        if '"sentiment"' in line:
            sentiment = float(line.split('"sentiment":')[1].split('}')[0])
    return date, hour, sentiment

def mergeTweet(tweet_collected, key, sentiment):
        if isinstance(sentiment, float):
            if  bool(key) and bool(sentiment):
                if key in tweet_collected:
                    tweet_collected[key][0] += sentiment
                    tweet_collected[key][1] += 1
                else:
                    tweet_collected[key] = [sentiment, 1]
        elif isinstance(sentiment, list):
            if  bool(key) and bool(sentiment):
                if key in tweet_collected:
                    tweet_collected[key][0] += sentiment[0]
                    tweet_collected[key][1] += sentiment[1]
                else:
                    tweet_collected[key] = sentiment
        return tweet_collected

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("result_path", type=str)
    return parser.parse_args()

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

file_size = os.path.getsize("twitter-1mb.json")
chunk_size = file_size / 4
chunk_start = int(chunk_size * rank)
count_exist = 0
count = 0
count_not = 0
tweet_dict = {}
missing_c = 0
missing_s = 0
time_begin = time.time()



with open("twitter-1mb.json", encoding='utf-8') as file:

    file.seek(chunk_start)
    count_processed_chunk = 0

    while count_processed_chunk <= chunk_size:

        #skip first line if rank is not 0 so we do not have any (the previous rank will read this line)
        if rank != 0:
            line = file.readline()
            count_processed_chunk += len(line.encode("utf-8"))
            continue

        count+= 1
        line = file.readline()
        if not line:
            break

        date, hour, sentiment = splitTweet(line)
        tweet_dict = mergeTweet(tweet_dict, date, sentiment)
        tweet_dict = mergeTweet(tweet_dict, hour, sentiment)
        count_processed_chunk += len(line.encode("utf-8"))
            
    line = file.readline()
    if line:
        date, hour, sentiment = splitTweet(line)
        tweet_dict = mergeTweet(tweet_dict, date, sentiment)
        tweet_dict = mergeTweet(tweet_dict, hour, sentiment)

gathered_list = comm.Gather(tweet_dict, root=0)
gathered_tweet = {}

if gathered_list is not None:
    for gathered_dict in gathered_list:
        for key, value_list in gathered_dict.items():
            gathered_tweet = mergeTweet(gathered_tweet, key, value_list)

# hour sentiment hour count date sentiment date count    

date_max_sentiment = {}
hour_max_sentiment = {}
date_max_count = {}
hour_max_count = {}


for key, value_list in gathered_tweet.items():
    if len(key) == 2:
        hour_max_sentiment[key] = value_list[0]
        hour_max_count[key] = value_list[1]
    else:
        date_max_sentiment[key] = value_list[0]
        date_max_count[key] = value_list[1]


key_date_sentiment, value_date_sentiment = maxFinder(date_max_sentiment)
key_date_count, value_date_count = maxFinder(date_max_count)
key_hour_sentiment, value_hour_sentiment = maxFinder(hour_max_sentiment)
key_hour_count, value_hour_count = maxFinder(hour_max_count)

if rank == 0:
        with open("./results/test4.txt", "w")  as f:
            time_end = time.time()
            print(f"Date with the highest tweet count is {key_date_count} with {value_date_count} tweets\n", file = f)
            print(f"Hour with the highest tweet count is {key_hour_count} with {value_hour_count} tweets\n", file = f)
            print(f"The happiest date is {key_date_sentiment} with {value_date_sentiment} sentiment\n",file = f)
            print(f"The happiest hour is {key_hour_sentiment} with {value_hour_sentiment} sentiment\n",file = f)
            print(f"Total run time = {time_end - time_begin}\n", file = f)
