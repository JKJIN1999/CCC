from mpi4py import MPI
import argparse
import os
import time


def main():
    time_begin = time.time()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    count = 0
    tweet_dict = {}
    
    args = getArgs()
    data_path = args.data_path
    result_path = args.result_path

    file_size = os.path.getsize(data_path)
    chunk_size = file_size / size
    chunk_start = int(chunk_size * rank)
    generated_from_cut = 0
    count_cut = 0

    with open(data_path, encoding='utf-8') as file:
        file.seek(chunk_start)
        count_processed_chunk = 0
        sentiment = None
        created_at = None
        tweet_id = None
        line_cut = True
        count = 0
        while count_processed_chunk <= chunk_size:
            count+= 1
            line = file.readline()
            if not line:
                break
            ## Search if the current line is cut line 
            if "}]}}," in line:
                line_cut = False
            else:
                print(f"this is line for line with problem : {line}")
                count_cut += 1
                line_cut = True
            if '"key":' in line and '"value"' in line:
                tweet_id = line.split('"key":')[1].split('"value"')[0]
            if "created_at" in line:
                created_at = line.split('"created_at":"')[1].split('"',1)[0].split(":")[0]
            if '"score"' not in line:
                if '"sentiment"' in line:
                    sentiment = line.split('"sentiment":')[1].split('}')[0]
            
            if bool(tweet_id) and bool(created_at) and bool(sentiment):
                tweet_dict[tweet_id] = [created_at,sentiment]
                if line_cut == True:
                    print("all found but was cut so we move on")
                line_cut = False
                sentiment = None
                created_at = None
                tweet_id = None

            count_processed_chunk += len(line.encode("utf-8"))
             
        if line_cut:
            #extend line and find other  possible missing values
            while True:
                print(f"started to look for next line ")
                line = file.readline()
                if not line:
                    break
                print(f"this is line for extend line : {line}")
                count += 1
                if not bool(tweet_id) and '"key":' in line and '"value"' in line:
                    tweet_id = line.split('"key":')[1].split('"value"')[0]
                if not bool(created_at) and "created_at" in line:
                    created_at = line.split('"created_at":"')[1].split('"',1)[0].split(":")[0]
                if '"score"' not in line:
                    if not bool(sentiment) and '"sentiment"' in line:
                        sentiment = line.split('"sentiment":')[1].split('}')[0]
                
                if bool(tweet_id) and bool(created_at) and bool(sentiment):
                    generated_from_cut += 1
                    tweet_dict[tweet_id] = [created_at,sentiment]
                    print(f"generated tweet dict : {created_at} and {sentiment}")
                    break
                else:
                    count_cut+= 1

    gathered_list = comm.gather(tweet_dict, root=0)

    date_dict = {}
    hour_dict = {}
    c_date_dict = {}
    c_hour_dict = {}

    result = []
    
    if gathered_list != None:
        print(len(gathered_list))
        for gathered_dict in gathered_list:
            print(len(gathered_dict))
            for collected_list in gathered_dict.values():
                created_at, sentiment = collected_list
                date, hour = created_at.split("T")
                hour = hour[:2]
                if date in date_dict:
                    date_dict[date] += float(sentiment)
                    c_date_dict[date] += 1
                else:
                    date_dict[date] = float(sentiment)
                    c_date_dict[date] = 1
                if hour in hour_dict:
                    hour_dict[hour] += float(sentiment) 
                    c_hour_dict[hour] += 1
                else:
                    hour_dict[hour] = float(sentiment)
                    c_hour_dict[hour] = 1

    for dict in (date_dict,hour_dict,c_date_dict,c_hour_dict):
        result.append(maxFinder(dict))
    
    if rank == 0:
        with open(result_path, "w")  as f:
            time_end = time.time()
            print(f"Date with the highest tweet count is {result[2][0]} with {result[2][1]} tweets\n", file = f)
            print(f"Hour with the highest tweet count is {result[3][0]} with {result[3][1]} tweets\n", file = f)
            print(f"The happiest date is {result[0][0]} with {result[0][1]} sentiment\n",file = f)
            print(f"The happiest hour is {result[1][0]} with {result[1][1]} sentiment\n",file = f)
            print(f"Total run time = {time_end - time_begin}\n", file = f)

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("result_path", type=str)
    return parser.parse_args()

def maxFinder(dic):
    if not dic:
        return dic
    max_key, max_value = max(dic.items(), key=lambda x: x[1])
    return [max_key, max_value]
    
 
if __name__ == "__main__":
    main()
    