import numpy as np
import pandas as pd
import category_encoders as ce


def load_data(args: dict) -> dict:

    if "browsing" in args.path:
        
        df = pd.read_csv(path, delimiter=",")
        
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
                if len(trace) >= MIN_TRACE_LEN:
                    if data[i - 1, 1] - start_time < MAX_TRACE_DURATION:
                        user_to_traces_map[int(data[i - 1, 0])].append(trace)
                trace = []
                start_time = data[i, 1]

            if len(trace) >= MAX_TRACE_LEN or (prev_time != 0.0 and data[i, 1] - prev_time > MAX_DELAY):
                if len(trace) >= MIN_TRACE_LEN and data[i - 1, 1] - start_time < MAX_TRACE_DURATION:
                    trace = pad_trace(trace)
                    user_to_traces_map[int(data[i, 0])].append(trace)
                trace = []
                start_time = data[i, 1]

            trace.append(data[i, 2:])
            prev_time = data[i, 1]
            prev_user = int(data[i, 0])

        del data    
        filtered_user_to_traces_map = {k: v for k, v in user_to_traces_map.items() if len(v) >= MIN_NUM_TRACES_PER_CLIENT}
        return filtered_user_to_traces_map
    
    else:
        return {}


def pad_trace(trace: list) -> list:
    if len(trace) < MAX_TRACE_LEN:
        diff = MAX_TRACE_LEN - len(trace)
        trace.extend([np.zeros((EMBEDDING_DIM)) for i in range(diff)])
    return trace
