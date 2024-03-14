import json
from mpi4py import MPI
import time
from collections import defaultdict

def mergeDictionary(gathered_dic):
    merged = defaultdict(int)
    if isinstance(gathered_dic, list):
        for dic in gathered_dic:
            for key in dic:
                if key in merged:
                    merged[key] += dic[key]
                else:
                    merged[key] = dic[key]
    elif isinstance(gathered_dic, dict):
        return gathered_dic
    return merged

def processJson(sca_data):
    date_sentiment = defaultdict(float)
    hour_sentiment = defaultdict(float)
    hour_count = defaultdict(int)
    date_count = defaultdict(int)

    for data_rows in sca_data:
        sentiment = 0
        date = None
        hour = None
        if "doc" in data_rows:
            doc = data_rows["doc"]
            if "data" in doc:
                item = doc["data"]
                if isinstance(item.get("sentiment"),(float,int)):
                    sentiment = item.get("sentiment")
                elif isinstance(item.get("sentiment", {}).get("score"),(float,int)):
                    sentiment = float(item.get("sentiment", {}).get("score"))
                date, hour = item["created_at"].split("T")
                hour = hour[:2]

                date_sentiment[date] += sentiment
                date_count[date] += 1

                hour_sentiment[hour] += sentiment
                hour_count[hour] += 1

    return dict(date_sentiment), dict(hour_sentiment), dict(date_count), dict(hour_count)

def maxFinder(dic):
    if not dic:
        return dic
    max_key, max_value = max(dic.items(), key=lambda x: x[1])
    return [max_key, max_value]

start_time = 0
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
data = None
time_begin = 0.0

merged_date = {}
merged_hour = {}
time_begin = time.time()

if rank == 0:
    with open('twitter-50mb.json') as file:
        data = json.load(file)["rows"]  
        data = [data[i::size] for i in range(size)]

sca_data = comm.scatter(data, root=0)

date_sentiment, hour_sentiment, date_count, hour_count = processJson(sca_data)

result = []

for data_type in (date_sentiment, hour_sentiment, date_count, hour_count):
    type_gathered = comm.gather(data_type, root=0)
    merged = mergeDictionary(type_gathered)
    result.append(maxFinder(merged))

if rank == 0:
    print(f"Date with the highest tweet count is {result[2][0]} with {result[2][1]} tweets")
    print(f"Hour with the highest tweet count is {result[3][0]} with {result[3][1]} tweets")
    print(f"The happiest date is {result[0][0]} with {result[0][1]} sentiment")
    print(f"The happiest hour is {result[1][0]} with {result[1][1]} sentiment")

    time_end = time.time()
    print(f"Total time taken = {time_end - time_begin}")