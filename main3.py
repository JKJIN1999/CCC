
from mpi4py import MPI
from collections import defaultdict
import orjson
import json
import argparse

def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    data = None
    time_begin = MPI.Wtime()
    
    args = getArgs()
    data_path = args.data_path
    result_path = args.result_path
    
    if rank == 0:
        with open(data_path, encoding='utf-8') as file:
            data = orjson.loads(file.read())["rows"]  
            data = [data[i::size] for i in range(size)]

    sca_data = comm.scatter(data, root=0)

    processed_dict = processJson(sca_data)


    #processed_dict = {day_time: [sentiment,count]}
    gathered_dict = comm.gather(processed_dict, root=0)

    date_dict = {}
    hour_dict = {}
    c_date_dict = {}
    c_hour_dict = {}

    result = []
    if gathered_dict is not None:
        for item in gathered_dict:
            for key, value in item.items():
                date, hour = key.split("T")
                if date in date_dict:
                    date_dict[date] = value[0] + date_dict.get(date)
                    c_date_dict[date] = value[1] + c_date_dict.get(date)
                else:
                    date_dict[date] = value[0]
                    c_date_dict[date] = value[1]
                if hour in hour_dict:
                    hour_dict[hour] = value[0] + hour_dict.get(hour)
                    c_hour_dict[hour] = value[1] + c_hour_dict.get(hour)
                else:
                    hour_dict[hour] = value[0]
                    c_hour_dict[hour] = value[1]

    for dict in (date_dict,hour_dict,c_date_dict,c_hour_dict):
        result.append(maxFinder(dict))
    
    if rank == 0:
        time_end = MPI.Wtime() - time_begin
        with open(result_path, "w")  as f:
            print(f"Date with the highest tweet count is {result[2][0]} with {result[2][1]} tweets\n", file = f)
            print(f"Hour with the highest tweet count is {result[3][0]} with {result[3][1]} tweets\n", file = f)
            print(f"The happiest date is {result[0][0]} with {result[0][1]} sentiment\n",file = f)
            print(f"The happiest hour is {result[1][0]} with {result[1][1]} sentiment\n",file = f)
            print(f"Total run time = {time_end - time_begin}\n", file = f)

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

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("result_path", type=str)
    return parser.parse_args()

#input scattered data and output date_hour 

def processJson(sca_data):
    processed_dict = {}

    for data_rows in sca_data:
        sentiment = 0
        value = []
        date_hour = None
        if "doc" in data_rows:
            doc = data_rows["doc"]
            if "data" in doc:
                item = doc["data"]
                if isinstance(item.get("sentiment"),(float,int)):
                    sentiment = item.get("sentiment")
                elif isinstance(item.get("sentiment", {}).get("score"),(float,int)):
                    sentiment = float(item.get("sentiment", {}).get("score"))
                date_hour = item["created_at"][:13]
                if date_hour in processed_dict:
                    value = [(sentiment + processed_dict.get(date_hour)[0]),(1 + processed_dict.get(date_hour)[1])]
                else:
                    value.append(sentiment)
                    value.append(1)
                processed_dict[date_hour] = value
    return processed_dict

def maxFinder(dic):
    if not dic:
        return dic
    max_key, max_value = max(dic.items(), key=lambda x: x[1])
    return [max_key, max_value]
 
if __name__ == "__main__":
    main()