import json
from mpi4py import MPI
import time

def mergeDictionary(gathered_dic):
    merged = {}
    if isinstance(gathered_dic, list):
        for dic in gathered_dic:
            for key, value in dic.items():
                merged[key] = merged.get(key, 0) + value
    elif isinstance(gathered_dic, dict):
        return gathered_dic
    return merged

def processJson(sca_data):
    date_sentiment = {}
    hour_sentiment = {}
    hour_count = {}
    date_count = {}

    for data_rows in sca_data:
        sentiment = 0
        if "doc" in data_rows:
            doc = data_rows["doc"]
            if "data" in doc:
                item = doc["data"]
                if isinstance(item.get("sentiment"),(float,int)):
                    sentiment = item.get("sentiment")
                else:
                    sentiment = item.get("sentiment", {}).get("score", 0)
                date, hour = item["created_at"].split("T")
                hour = hour[:2]

                date_sentiment[date] = date_sentiment.get(date, 0) + sentiment
                date_count[date] = date_count.get(date, 0) + 1

                hour_sentiment[hour] = hour_sentiment.get(hour, 0) + sentiment
                hour_count[hour] = hour_count.get(hour, 0) + 1

    return date_sentiment, hour_sentiment, date_count, hour_count

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
    print(f"Date with highest tweet count is {result[2][0]} with {result[2][1]} tweets")
    print(f"Hour with highest tweet count is {result[3][0]} with {result[3][1]} tweets")
    print(f"Happiest date is {result[0][0]} with {result[0][1]} sentiment")
    print(f"Happiest hour is {result[1][0]} with {result[1][1]} sentiment")

    time_end = time.time()
    print(f"Total time taken = {time_end - time_begin}")