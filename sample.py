import tensorflow as tf
import numpy as np
import random


def get_random_triplet_batch(args: dict, user_to_traces_map: dict, batch_size=None):
    """Generates a random triplet training batch."""
    batch_anchor = []
    batch_positive = []
    batch_negative = []
    batch_labels = []

    if batch_size is None:
        batch_size = args.batch_size

    for i in range(batch_size):
        
        random_users_list = random.sample(user_to_traces_map.keys(), 2)
        anchor_user = random_users_list[0]
        traces_list = user_to_traces_map[anchor_user]
        sampled_traces = random.sample(traces_list, 2)
        anchor_trace = sampled_traces[0]
        positive_trace = sampled_traces[1]

        negative_user = random_users_list[1]
        traces_list = user_to_traces_map[negative_user]
        negative_trace = random.sample(traces_list, 1)[0]

        batch_labels.append(0)
        batch_anchor.append(anchor_trace)
        batch_positive.append(positive_trace)
        batch_negative.append(negative_trace)

    return [np.asarray(batch_anchor, dtype='float32'), np.asarray(batch_positive, dtype='float32'), np.asarray(batch_negative, dtype='float32')], np.asarray(batch_labels)


def get_hard_triplet_batch(args: dict, user_to_traces_map: dict, model: tf.keras.Model):
    """Generates a hard triplet training batch."""

    batch_size = args.batch_size * 2
    hard_batch_size = int(batch_size / 2)
    
    random_batch, random_labels = get_random_triplet_batch(args, user_to_traces_map, batch_size)
    random_anchor_batch, random_positive_batch, random_negative_batch = random_batch

    output = model.predict(random_batch)
    anchor, positive, negative = output[:, :args.latent_size], output[: ,args.latent_size:2*args.latent_size], output[:, 2*args.latent_size:]
    dist = tf.reduce_mean(tf.square(anchor - positive), axis=1) - tf.reduce_mean(tf.square(anchor - negative), axis=1)
    selection = tf.argsort(dist, direction="DESCENDING")[:hard_batch_size]

    return [random_anchor_batch[selection], random_positive_batch[selection], random_negative_batch[selection]], random_labels[selection]


def get_semi_hard_triplet_batch(args: dict, user_to_traces_map: dict, model: tf.keras.Model):
    """Generates a semi hard triplet training batch."""

    batch_size = args.batch_size * 2

    hard_batch_size = int(batch_size / 4)
    norm_batch_size = int(batch_size / 4)
    
    random_batch, random_labels = get_random_triplet_batch(args, user_to_traces_map, batch_size)
    random_anchor_batch, random_positive_batch, random_negative_batch = random_batch

    output = model.predict(random_batch)
    anchor, positive, negative = output[:, :args.latent_size], output[: ,args.latent_size:2*args.latent_size], output[:, 2*args.latent_size:]
    dist = tf.reduce_mean(tf.square(anchor - positive), axis=1) - tf.reduce_mean(tf.square(anchor - negative), axis=1)
    selection_hard = tf.argsort(dist, direction="DESCENDING")[:hard_batch_size]

    # Select other random samples
    selection_norm = np.random.choice(np.delete(np.arange(batch_size),selection_hard), norm_batch_size,replace=False)
    selection = np.append(selection_hard, selection_norm)
    
    return [random_anchor_batch[selection], random_positive_batch[selection], random_negative_batch[selection]], random_labels[selection]


def get_hard_pair_batch(args: dict, user_to_traces_map: dict, model: tf.keras.Model):
    """Generates a hard pair training batch."""
    
    batch_size = args.batch_size * 2
    hard_batch_size = int(batch_size / 2)

    random_batch, random_labels = get_random_pair_batch(args, user_to_traces_map, batch_size)
    random_batch_x1, random_batch_x2 = random_batch

    output = model.predict(random_batch)
    dist = np.squeeze(output)
    selection = tf.argsort(dist, direction="DESCENDING")[:hard_batch_size] 

    return [random_batch_x1[selection], random_batch_x2[selection]], random_labels[selection]


def get_semi_hard_pair_batch(args: dict, user_to_traces_map: dict, model: tf.keras.Model):
    """Generates a semi-hard pair training batch."""
    
    batch_size = args.batch_size * 2

    hard_batch_size = int(batch_size / 4)
    norm_batch_size = int(batch_size / 4)
    
    random_batch, random_labels = get_random_pair_batch(args, user_to_traces_map, batch_size)
    random_batch_x1, random_batch_x2 = random_batch

    output = model.predict(random_batch)
    dist = np.squeeze(output)
    selection_hard = tf.argsort(dist, direction="DESCENDING")[:hard_batch_size] 

    # Select other random samples
    selection_norm = np.random.choice(np.delete(np.arange(batch_size),selection_hard), norm_batch_size, replace=False)
    selection = np.append(selection_hard, selection_norm)

    return [random_batch_x1[selection], random_batch_x2[selection]], random_labels[selection]



def get_random_pair_batch(args: dict, user_to_traces_map: dict, batch_size=None):
    """Generates a random pair training batch."""

    batch_x1 = []
    batch_x2 = []
    batch_labels = []

    if batch_size is None:
        batch_size = args.batch_size

    for i in range(int(batch_size / 2)):
        random_user = random.sample(user_to_traces_map.keys(), 1)[0]
        traces_list = user_to_traces_map[random_user]
        sampled_traces = random.sample(traces_list, 2)
        trace_1 = sampled_traces[0]
        trace_2 = sampled_traces[1]
        batch_x1.append(trace_1)
        batch_x2.append(trace_2)
        batch_labels.append(1)
    
    for i in range(int(batch_size / 2)):
        random_users = random.sample(user_to_traces_map.keys(), 2)
        traces_list_1 = user_to_traces_map[random_users[0]]
        traces_list_2 = user_to_traces_map[random_users[1]]
        trace_1 = random.sample(traces_list_1, 1)[0]
        trace_2 = random.sample(traces_list_2, 1)[0]
        batch_x1.append(trace_1)
        batch_x2.append(trace_2)
        batch_labels.append(0)

    collection = list(zip(batch_x1, batch_x2, batch_labels))
    random.shuffle(collection)
    batch_x1, batch_x2, batch_labels = zip(*collection)

    return [np.asarray(batch_x1, dtype='float32'), np.asarray(batch_x2, dtype='float32')], np.asarray(batch_labels)


def get_test_pair_batch(args: dict, user_to_traces_map: dict, user_to_test_idx_map: dict, target_trace: list, target_user: int, p: int):
    """Generates a pair batch for testing."""

    batch_x1 = []
    batch_x2 = []
    batch_labels = []

    users = user_to_traces_map.keys()

    for i in range(p, p + args.batch_size):
        if i >= len(users):
            break
        test_user = list(users)[i]
        traces_list = user_to_traces_map[test_user]
        test_trace_idx = user_to_test_idx_map.get(test_user)
        test_trace = traces_list[test_trace_idx]
        if test_user == target_user:
            label = 1
        else:
            label = 0
        batch_x1.append(target_trace) 
        batch_x2.append(test_trace)
        batch_labels.append(label)

    return [np.asarray(batch_x1, dtype='float32'), np.asarray(batch_x2, dtype='float32')], np.asarray(batch_labels)


def get_test_triplet_batch(args: dict, user_to_traces_map: dict, user_to_test_idx_map: dict, target_trace: list, target_user: int, p: int):
    """Generates a triplet batch for testing. 

        Params:
            - args: dict - dictionary keeping all parameter arguments
            - user_to_traces_map: dict: 
    
    """
    
    batch_anchor = []
    batch_positive = []
    batch_negative = []
    batch_labels = []

    users = list(user_to_traces_map.keys())
    
    for i in range(p, p + args.batch_size):
        if i >= len(users):
            break
        test_user = users[i]
        traces_list = user_to_traces_map[test_user]
        test_trace_idx = user_to_test_idx_map.get(test_user)
        test_trace = traces_list[test_trace_idx]
        if test_user == target_user:
            label = 1
        else:
            label = 0

        users_without_target_user = users.copy()
        if target_user in users_without_target_user:
            users_without_target_user.remove(target_user)
        negative_user = random.sample(users_without_target_user, 1)[0]
        traces_list = user_to_traces_map[negative_user]
        split = int(len(traces_list) / 2)
        negative_trace = random.sample(traces_list[:split], 1)[0]

        batch_anchor.append(target_trace) 
        batch_positive.append(test_trace)
        batch_negative.append(negative_trace)
        batch_labels.append(label) 

    return [np.asarray(batch_anchor, dtype='float32'), np.asarray(batch_positive, dtype='float32'), np.asarray(batch_negative, dtype='float32')], np.asarray(batch_labels)
