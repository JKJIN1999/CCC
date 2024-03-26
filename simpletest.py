from mpi4py import MPI
import argparse
import os


def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    count = 0
    tweet_list = []
    args = getArgs()
    data_path = args.data_path
    result_path = args.result_path

    file_size = os.path.getsize(data_path)
    chunk_size = file_size / size
    chunk_start = int(chunk_size * rank)

    with open(data_path, encoding='utf-8') as file:
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
            if '"score"' not in line:
                if '"sentiment"' in line:
                    sentiment = line.split('"sentiment":')[1].split('}')[0] 
            
            if bool(created_at) and bool(sentiment):
                count += 1
                tweet_list.append([created_at, sentiment])
        count_processed_chunk += len(line.encode("utf-8"))

    gathered_list = comm.gather(tweet_list, root=0)

    date_dict = {}
    hour_dict = {}
    c_date_dict = {}
    c_hour_dict = {}

    result = []

    if gathered_list is not None:
        for item in gathered_list:
            created_at, sentiment = item
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
            time_end = MPI.Wtime() - time_begin
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
    time_begin = MPI.Wtime()
    main()
    