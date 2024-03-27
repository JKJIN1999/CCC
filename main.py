from mpi4py import MPI
import argparse
import os
import time
import sys

def maxFinder(dic):
    if not dic:
        return None, None
    else:
        max_key, max_value = max(dic.items(), key=lambda x: x[1])
        return max_key, max_value

def splitTweet(line):
    created_at = None
    sentiment = None
    if "created_at" in line:
            created_at = line.split('"created_at":"')[1].split('"',1)[0].split(":")[0]
    if '"score"' not in line:
        if '"sentiment"' in line:
            sentiment = float(line.split('"sentiment":')[1].split('}')[0])
    return created_at, sentiment

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

def main():
    time_begin = time.time()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    tweet_dict = {}
    
    args = getArgs()
    data_path = args.data_path
    result_path = args.result_path

    file_size = os.path.getsize(data_path)
    chunk_size = file_size / size
    chunk_start = int(chunk_size * rank)

    

    with open(data_path, encoding='utf-8') as file:

        file.seek(chunk_start)
        count_processed_chunk = 0
        if rank > 0:
            line = file.readline()
            count_processed_chunk += len(line.encode("utf-8"))

        while count_processed_chunk <= chunk_size:

            #skip first line if rank is not 0 so we do not have any (the previous rank will read this line)
            line = file.readline()
            if not line:
                break
            created_at, sentiment = splitTweet(line)
            tweet_dict = mergeTweet(tweet_dict, created_at, sentiment)
            count_processed_chunk += len(line.encode("utf-8"))
             
        
        #this line is the first line of the next rank

        line = file.readline()
        created_at, sentiment = splitTweet(line)
        tweet_dict = mergeTweet(tweet_dict, created_at, sentiment)

    gathered_list = comm.gather(tweet_dict, root=0)
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
        hour_max_sentiment[key] = value_list[0]
        hour_max_count[key] = value_list[1]
        date_key = key.split("T")[0]
        print(date_key)
        date_max_sentiment[date_key] = value_list[0]
        date_max_count[date_key] = value_list[1]
            
    key_date_sentiment, value_date_sentiment = maxFinder(date_max_sentiment)
    key_date_count, value_date_count = maxFinder(date_max_count)
    key_hour_sentiment, value_hour_sentiment = maxFinder(hour_max_sentiment)
    key_hour_count, value_hour_count = maxFinder(hour_max_count)
    
    if rank == 0:
        with open(result_path, "w")  as f:
            time_end = time.time()
            print(f"Date with the highest tweet count is {key_date_count} with {value_date_count} tweets\n", file = f)
            print(f"Hour with the highest tweet count is {key_hour_count} with {value_hour_count} tweets\n", file = f)
            print(f"The happiest date is {key_date_sentiment} with {value_date_sentiment} sentiment\n",file = f)
            print(f"The happiest hour is {key_hour_sentiment} with {value_hour_sentiment} sentiment\n",file = f)
            print(f"Total run time = {time_end - time_begin}\n", file = f)
    comm.barrier()
if __name__ == "__main__":
    sys.exit(main())