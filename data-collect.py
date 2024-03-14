import json
from mpi4py import MPI
import time as clock_time



def mergeDictionary(gathered_dic):
    merged = {}
    if isinstance(gathered_dic, list):
        for dic in gathered_dic:
            for key, value in dic.items():
                if key not in merged:
                    merged[key] = value
                else:
                    if key in merged:
                        merged[key] += value 
            # merged = {**merged, **dic}
            # for key in merged:
            #     if key in merged and key in dic:
            #             merged[key] += dic[key]
    else:
        if isinstance(gathered_dic, dict):
            return gathered_dic
    return merged

 
date_sentiment = {}
hour_sentiment = {}
hour_count = {}
date_count = {}       

def processJson(sca_data):
    sentiment = 0
    for data_rows in sca_data:
        if "doc" in data_rows:
            doc = data_rows["doc"]
            if "data" in doc:
                item = doc["data"]
                if "sentiment" in item:
                    if isinstance(item["sentiment"],(float, int)):
                        sentiment = item["sentiment"]
                    else:
                        if isinstance(item["sentiment"]["score"],(float,int)) :
                            sentiment = item["sentiment"]["score"]
                        else:
                            print("No sentiment found")
                date, hour = item["created_at"].split("T")
                hour = hour[:2]
                
                if date in date_sentiment:
                    date_sentiment[str(date)] += sentiment
                    date_count[str(date)] += 1
                else:
                    date_sentiment[str(date)] = sentiment
                    date_count[str(date)] = 1  
                                            
                if hour in hour_sentiment:
                    hour_sentiment[str(hour)] += sentiment
                    hour_count[str(hour)] += 1
                else:
                    hour_sentiment[str(hour)] = sentiment
                    hour_count[str(hour)] = 1 
            
# [hour_dic(key:[0,1]) ,...]
#                S C
def maxFinder(dic):
    max_value = None
    max_key = None
    dic_len = len(dic)
    if isinstance(dic,dict) and dic_len > 1:
        for key in dic:
            if max_value is None:
                max_value = dic[key]
                max_key = key
            elif max_value < dic[key]:
                max_value = dic[key]
                max_key = key
    elif dic_len == 1:
        max_key = list(dic.keys())[0]
        max_value = dic[max_key]
    else:
        return dic
    return [max_key, max_value]
 
start_time = 0
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
data = None
time_begin = 0.0

merged_date = {}
merged_hour = {}
time_begin = clock_time.time()

if rank == 0:
    data = json.load(open('twitter-50mb.json'))["rows"]
    data = [data[i::size] for i in range(size)]
    
    
sca_data = comm.scatter(data, root=0)

processJson(sca_data)

result = []

for type in (date_sentiment, hour_sentiment, date_count, hour_count):
    type = comm.gather(type, root=0)
    merged = mergeDictionary(type)
    max = maxFinder(merged)
    result.append(maxFinder(max))


print(f"Date with highest tweet count is  {result[2][0]} with {result[2][1]} tweets\n")
print(f"Hour with highest tweet count is  {result[3][0]} with {result[3][1]} tweets\n")
print(f"Happiest date is {result[0][0]} with {result[0][1]} sentiment\n")
print(f"Happiest hour is {result[1][0]} with {result[1][1]} sentiment\n")

time_end = clock_time.time()
print (f"Total time taken = {(time_end) - (time_begin)}")