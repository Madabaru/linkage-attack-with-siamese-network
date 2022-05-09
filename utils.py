import numpy as np
import pandas as pd
import category_encoders as ce
import random
import logging
import pickle
import collections

from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict

def load_data(args: dict) -> dict:
    """Loads the data and transforms it."""

    if "browsing" in args.path:
       
        df = pd.read_csv(args.path, delimiter=",") 

        df["date"] = pd.to_datetime(df["timestamp"], unit="s")
        df["weekday"] = df["date"].dt.weekday
        df["hour"] = df["date"].dt.hour

        weekday_encoder = ce.BinaryEncoder(cols="weekday", return_df=False)
        weekday_encoded = weekday_encoder.fit_transform(df["weekday"])
        hour_encoder = ce.BinaryEncoder(cols="hour", return_df=False)
        hour_encoded = hour_encoder.fit_transform(df["hour"])
        
        gender_encoder = ce.OneHotEncoder(cols="gender", return_df=False)
        gender_encoded = gender_encoder.fit_transform(df["gender"])
        age_encoder = ce.OneHotEncoder(cols="age", return_df=False)
        age_encoded = age_encoder.fit_transform(df["age"])
        url_encoder = ce.BinaryEncoder(cols="url", return_df=False)
        url_encoded = url_encoder.fit_transform(df["url"])
        cat_encoder = ce.BinaryEncoder(cols="category", return_df=False)
        cat_encoded = cat_encoder.fit_transform(df["category"])
        domain_encoder = ce.BinaryEncoder(cols="domain", return_df=False)
        domain_encoded = domain_encoder.fit_transform(df["domain"])

        user_id = df["user_id"].values
        timestamp = df["timestamp"].values

        data = np.column_stack((user_id, timestamp, gender_encoded, url_encoded, cat_encoded, domain_encoded, age_encoded)) 
        embedding_dim = data.shape[1]
        args.embedding_dim = embedding_dim - 2
        logging.info("Embedding Dimesion: %i", args.embedding_dim) 

        prev_time = 0.0
        start_time = data[0, 1]
        trace_lengths = []

        user_to_traces_map = OrderedDict()
        trace = []

        for i in range(0, data.shape[0]):
            if not int(data[i, 0]) in user_to_traces_map: 
                user_to_traces_map[int(data[i, 0])] = []
                if len(trace) >= args.min_trace_len:
                    if data[i - 1, 1] - start_time <= args.max_trace_duration:
                        trace_lengths.append(len(trace))
                        trace = pad_trace(args, trace)
                        user_to_traces_map[int(data[i - 1, 0])].append(trace)
                trace = []
                start_time = data[i, 1]
            if len(trace) >= args.max_trace_len or (prev_time != 0.0 and data[i, 1] - prev_time >= args.max_delay):
                if len(trace) >= args.min_trace_len and data[i - 1, 1] - start_time <= args.max_trace_duration:
                    trace = pad_trace(args, trace)
                    trace_lengths.append(len(trace))
                    user_to_traces_map[int(data[i, 0])].append(trace)
                trace = []
                start_time = data[i, 1]

            trace.append(data[i, 2:])
            prev_time = data[i, 1]
            prev_user = int(data[i, 0])

        trace = pad_trace(args, trace)
        trace_lengths.append(len(trace))
        user_to_traces_map[int(data[-1, 0])].append(trace)
        logging.info("Average trace length: %i", np.mean(trace_lengths)) 
        
        del data    
        user_to_traces_map = {i: v[1] for i, v in enumerate(user_to_traces_map.items())}

        logging.info("Number of users before: %i", len(user_to_traces_map.keys())) 
        filtered_user_to_traces_map = {k: v for k, v in user_to_traces_map.items() if len(v) >= args.min_num_traces_per_user} 
        logging.info("Number of users after: %i", len(filtered_user_to_traces_map.keys())) 
        num_samples = sum([len(v) for k, v in filtered_user_to_traces_map.items()])
        logging.info("Total number of traces: %i", num_samples) 

        return filtered_user_to_traces_map
    
    else:

        df = pd.read_csv(args.path, delimiter=",")

        df["date"] = pd.to_datetime(df["timestamp"], unit="s")
        df["weekday"] = df["date"].dt.weekday
        df["hour"] = df["date"].dt.hour

        weekday_encoder = ce.BinaryEncoder(cols="weekday", return_df=False)
        weekday_encoded = weekday_encoder.fit_transform(df["weekday"])
        hour_encoder = ce.BinaryEncoder(cols="hour", return_df=False)
        hour_encoded = hour_encoder.fit_transform(df["hour"])
        street_encoder = ce.BinaryEncoder(cols="street", return_df=False)
        street_encoded = street_encoder.fit_transform(df["street"])
        postcode_encoder = ce.BinaryEncoder(cols="postcode", return_df=False)
        postcode_encoded = postcode_encoder.fit_transform(df["postcode"])
        state_encoder = ce.OneHotEncoder(cols="state", return_df=False)
        state_encoded = state_encoder.fit_transform(df["state"])
        highway_encoder = ce.BinaryEncoder(cols="highway", return_df=False)
        highway_encoded = highway_encoder.fit_transform(df["highway"])
        hamlet_encoder = ce.BinaryEncoder(cols="hamlet", return_df=False)
        hamlet_encoded = hamlet_encoder.fit_transform(df["hamlet"])
        suburb_encoder = ce.BinaryEncoder(cols="suburb", return_df=False)
        suburb_encoded = suburb_encoder.fit_transform(df["suburb"])
        village_encoder = ce.BinaryEncoder(cols="village", return_df=False)
        village_encoded = village_encoder.fit_transform(df["village"])
        location_code_encoder = ce.BinaryEncoder(cols="location_code", return_df=False)
        location_code_encoded = location_code_encoder.fit_transform(df["location_code"])

        df["speed"] = MinMaxScaler().fit_transform(df["speed"].values.reshape(-1,1))
        df["heading"] = MinMaxScaler().fit_transform(df["heading"].values.reshape(-1,1))

        client_id = df["client_id"].values
        timestamp = df["timestamp"].values
        speed = df["speed"].values
        heading = df["heading"].values

        fields = [client_id, timestamp]
        if "state" in args.fields:
            fields.append(state_encoded)
        if "street" in args.fields:
            fields.append(street_encoded)
        if "postcode" in args.fields:
            fields.append(postcode_encoded)
        if "location_code" in args.fields:
            fields.append(location_code_encoded)
        if "hour" in args.fields:
            fields.append(hour_encoded)
        if "weekday" in args.fields:
            fields.append(weekday_encoded)
        if "highway" in args.fields:
            fields.append(highway_encoded)
        if "hamlet" in args.fields:
            fields.append(hamlet_encoded)
        if "village" in args.fields:
            fields.append(village_encoded)
        if "suburb" in args.fields:
            fields.append(suburb_encoded)
        if "speed" in args.fields:
            fields.append(speed)
        if "heading" in args.fields:
            fields.append(heading)

        data = np.column_stack(fields)
        embedding_dim = data.shape[1]
        args.embedding_dim = embedding_dim - 2
        logging.info("Embedding Dimesion: %i", args.embedding_dim) 

        prev_time = 0.0
        start_time = data[0, 1]

        user_to_traces_map = OrderedDict()
        trace_lengths = []
        trace = []
        trace_coordinates = []

        for i in range(0, data.shape[0]):

            if not int(data[i, 0]) in user_to_traces_map: 
                user_to_traces_map[int(data[i, 0])] = []
                if len(trace) >= args.min_trace_len:
                    if data[i - 1, 1] - start_time <= args.max_trace_duration:
                        trace_lengths.append(len(trace))
                        trace = pad_trace(args, trace)
                        user_to_traces_map[int(data[i - 1, 0])].append(trace)
                trace = []
                start_time = data[i, 1]

            if len(trace) >= args.max_trace_len or (prev_time != 0.0 and data[i, 1] - prev_time >= args.max_delay):
                if len(trace) >= args.min_trace_len and data[i - 1, 1] - start_time <= args.max_trace_duration:
                    trace_lengths.append(len(trace))
                    trace = pad_trace(args, trace)
                    user_to_traces_map[int(data[i, 0])].append(trace)
                trace = []
                start_time = data[i, 1]

            trace.append(data[i, 2:])
            prev_time = data[i, 1]
            prev_user = int(data[i, 0])

        del data    
        user_to_traces_map = {i: v[1] for i, v in enumerate(user_to_traces_map.items())}
        
        logging.info("Average trace length: %i", np.mean(trace_lengths))         
        logging.info("Number of clients before: %i", len(user_to_traces_map.keys())) # Rust 7933
        filtered_user_to_traces_map = {k: v for k, v in user_to_traces_map.items() if len(v) >= args.min_num_traces_per_user} 
        logging.info("Number of clients after: %i", len(filtered_user_to_traces_map.keys())) # Rust: 3861
        num_samples = sum([len(v) for k, v in filtered_user_to_traces_map.items()])
        logging.info("Total number of traces: %i", num_samples) # Rust: 130756 
        return filtered_user_to_traces_map


def pad_trace(args: dict, trace: list) -> list:
    """Adds padding a given trace."""
    if len(trace) < args.max_trace_len:
        diff = args.max_trace_len - len(trace)
        trace.extend([np.zeros((args.embedding_dim)) for i in range(diff)])
    return trace


def gen_user_to_target_idx_map(args: dict, user_to_traces_map: dict):
    """Generates a dictionary with random target traces indices (value) for each user (key)."""
    user_to_target_idx_map = {}
    target_user_list = random.sample(user_to_traces_map.keys(), args.sample_size)
    for target_user in target_user_list:
        traces_list = user_to_traces_map.get(target_user)
        start_idx = len(traces_list[int(len(traces_list) / 2):])
        end_idx = len(traces_list)
        target_index_list = list(range(start_idx, end_idx))
        if len(target_index_list) < args.num_target_traces:
            sampled_target_idx_list = random.sample(target_index_list, len(target_index_list))
        else:
            sampled_target_idx_list = random.sample(target_index_list, args.num_target_traces)
        user_to_target_idx_map[target_user] = sampled_target_idx_list
    return user_to_target_idx_map


def gen_user_to_sample_idx_map(user_to_traces_map: dict, k: int):
    """Generates a dictionary with random observed trace indicies (value) for each user (key)."""
    user_to_sample_idx_map = {}
    for client, traces_list in user_to_traces_map.items():
        if k == 0:
            user_to_sample_idx_map[client] = list(range(len(traces_list[:int(len(traces_list) / 2)])))
        else:
            if k > int(len(traces_list) / 2):
                user_to_sample_idx_map[client] = list(range(len(traces_list[:int(len(traces_list) / 2)])))
            else:
                user_to_sample_idx_map[client] = list(range(len(traces_list[:k])))
    return user_to_sample_idx_map


def gen_user_to_test_idx_map(user_to_sample_idx_map: dict):
    """Generates a dictionary with a single random observed trace index (value) for each user (key)."""
    user_to_test_idx_map = {client: random.sample(traces_list, 1)[0] for client, traces_list in user_to_sample_idx_map.items()}
    return user_to_test_idx_map
