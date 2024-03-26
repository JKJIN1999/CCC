
from mpi4py import MPI
from collections import defaultdict
import argparse
import os
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

file_size = os.path.getsize("twitter-50mb.json")
chunk_size = file_size / size

chunk_start = int(chunk_size * rank)
count_exist = 0
count = 0
count_not = 0
tweet_dict = {}
tweet_list = []
missing_c = 0
missing_s = 0
start_time = time.time()
with open("twitter-50mb.json", encoding='utf-8') as file:
    file.seek(chunk_start)
    count_processed_chunk = 0
    while count_processed_chunk <= chunk_size:
        sentiment = None
        created_at = None
        line = file.readline()
        if not line:
            break
        if "created_at" in line:
            created_at = line.split('"created_at":"')[1].split('"',1)[0].split(":")[0]
        else: missing_c +=1
        if '"score"' not in line:
            if '"sentiment"' in line:
                sentiment = line.split('"sentiment":')[1].split('}')[0]
            else: missing_s += 1                                         
        else: missing_s += 1
            
        if bool(created_at) and bool(sentiment):
            count += 1
            if created_at in tweet_dict:
                tweet_dict[created_at][0] += sentiment
                tweet_dict[created_at][1] += 1
            else:
                tweet_dict[created_at] = [sentiment, 1]
        
        count_processed_chunk += len(line.encode("utf-8"))
tweet_list.append(tweet_dict)
gathered_dict = comm.gather(tweet_list, root=0)
print(gathered_dict)
# print(count_exist)
# print(gathered_dict)
# print(len(gathered_dict))
# print(count)
# print(missing_c + missing_s)
# print(time.time() - start_time)