import numpy as np
import pandas as pd
import category_encoders as ce
import random


def load_data(args: dict) -> dict:
    """ Loads the data and transforms it. """

    if "browsing" in args.path:
        
        df = pd.read_csv(args.path, delimiter=",")
        
        age_encoder = ce.OneHotEncoder(cols="age", return_df=False)
        age_encoded = age_encoder.fit_transform(df["age"])
        url_encoder = ce.BinaryEncoder(cols="url", return_df=False)
        url_encoded = url_encoder.fit_transform(df["url"])
        cat_encoder = ce.BinaryEncoder(cols="category", return_df=False)
        cat_encoded = cat_encoder.fit_transform(df["category"])
        domain_encoder = ce.BinaryEncoder(cols="domain", return_df=False)
        domain_encoded = domain_encoder.fit_transform(df["domain"])

        user_id = df["user_id"].values
        gender = df["gender"].values
        timestamp = df["timestamp"].values

        data = np.column_stack((user_id, timestamp, gender, url_encoded, cat_encoded, domain_encoded, age_encoded))
        embedding_dim = data.shape[1]
        args.embedding_dim = embedding_dim - 2

        prev_time = 0.0
        start_time = data[0, 1]

        user_to_traces_map = {}
        trace = []

        for i in range(0, data.shape[0]):

            if not int(data[i, 0]) in user_to_traces_map: 
                user_to_traces_map[int(data[i, 0])] = []
                if len(trace) >= args.min_trace_len:
                    if data[i - 1, 1] - start_time < args.max_trace_duration:
                        user_to_traces_map[int(data[i - 1, 0])].append(trace)
                trace = []
                start_time = data[i, 1]

            if len(trace) >= args.max_trace_len or (prev_time != 0.0 and data[i, 1] - prev_time > args.max_delay):
                if len(trace) >= args.min_trace_len and data[i - 1, 1] - start_time < args.max_trace_duration:
                    trace = pad_trace(args, trace)
                    user_to_traces_map[int(data[i, 0])].append(trace)
                trace = []
                start_time = data[i, 1]

            trace.append(data[i, 2:])
            prev_time = data[i, 1]
            prev_user = int(data[i, 0])

        del data    
        filtered_user_to_traces_map = {k: v for k, v in user_to_traces_map.items() if len(v) >= args.min_num_traces_per_user}
        return filtered_user_to_traces_map
    
    else:
        return {}


def pad_trace(args: dict, trace: list) -> list:
    """ Adds padding a given trace. """
    if len(trace) < args.max_trace_len:
        diff = args.max_trace_len - len(trace)
        trace.extend([np.zeros((args.embedding_dim)) for i in range(diff)])
    return trace


def gen_target_user_to_target_trace_map(user_to_traces_map, target_user_list):
    """ Generates a map with target user as key and target traces as value. """
    target_user_to_target_trace_map = {}
    for target_user in target_user_list:
        traces_list = user_to_traces_map[target_user]
        split = int(len(traces_list) / 2)
        target_user_to_target_trace_map[target_user] = random.sample(traces_list[split:], 1)[0]
    return target_user_to_target_trace_map
