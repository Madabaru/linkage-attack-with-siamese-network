import tensorflow as tf
import numpy as np
import random


def get_random_triplet_batch(args: dict, user_to_traces_map: dict, batch_size=None):
    """
    Generates a random triplet training batch.

    Parameters:
        args (dict): dictionary of arguments
        user_to_traces_map (dict): dictionary that holds for each user (key) all associated traces (value)
        batch_size (int): size of the training batch

    Returns:
        triplet_training_batch (list): list of numpy arrays
        batch_labels (np.array): array of labels 
    """
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

        triplet_training_batch = [np.asarray(batch_anchor, dtype='float32'), np.asarray(
            batch_positive, dtype='float32'), np.asarray(batch_negative, dtype='float32')]
        batch_labels = np.asarray(batch_labels)

    return triplet_training_batch, batch_labels


def get_hard_triplet_batch(args: dict, user_to_traces_map: dict, model: tf.keras.Model):
    """
    Generates a hard triplet training batch.

    Parameters:
        args (dict): dictionary of arguments
        user_to_traces_map (dict): dictionary that holds for each user (key) all associated traces (value)
         model (tf.keras.Model): neural network model to perform a single forward pass 

    Returns:
        triplet_training_batch (list): list of numpy arrays
        batch_labels (np.array): array of labels 
    """
    batch_size = args.batch_size * 2
    hard_batch_size = int(batch_size / 2)

    random_batch, random_labels = get_random_triplet_batch(
        args, user_to_traces_map, batch_size)
    random_anchor_batch, random_positive_batch, random_negative_batch = random_batch

    # Perform the forward pass
    output = model.predict(random_batch)
    anchor, positive, negative = output[:, :args.latent_size], output[:, args.latent_size:2*args.latent_size], output[:, 2*args.latent_size:]
    dist = tf.reduce_mean(tf.square(anchor - positive), axis=1) - \
        tf.reduce_mean(tf.square(anchor - negative), axis=1)
    # Select samples that show the highest distance
    selection = tf.argsort(dist, direction="DESCENDING")[:hard_batch_size]

    triplet_training_batch = [random_anchor_batch[selection],
                              random_positive_batch[selection], random_negative_batch[selection]]
    batch_labels = random_labels[selection]

    return triplet_training_batch, batch_labels


def get_semi_hard_triplet_batch(args: dict, user_to_traces_map: dict, model: tf.keras.Model):
    """
    Generates a semi hard triplet training batch.

    Parameters:
        args (dict): dictionary of arguments
        user_to_traces_map (dict): dictionary that holds for each user (key) all associated traces (value)
        model (tf.keras.Model): neural network model to perform a single forward pass 

    Returns:
        triplet_training_batch (list): list of numpy arrays
        batch_labels (array): array of labels 
    """
    batch_size = args.batch_size * 2

    hard_batch_size = int(batch_size / 4)
    norm_batch_size = int(batch_size / 4)

    random_batch, random_labels = get_random_triplet_batch(
        args, user_to_traces_map, batch_size)
    random_anchor_batch, random_positive_batch, random_negative_batch = random_batch

    # Perform the forward pass
    output = model.predict(random_batch)
    anchor, positive, negative = output[:, :args.latent_size], output[:, args.latent_size:2*args.latent_size], output[:, 2*args.latent_size:]
    # Select samples that show the highest distance
    dist = tf.reduce_mean(tf.square(anchor - positive), axis=1) - \
        tf.reduce_mean(tf.square(anchor - negative), axis=1)
    selection_hard = tf.argsort(dist, direction="DESCENDING")[:hard_batch_size]

    # Select other random samples
    selection_norm = np.random.choice(np.delete(
        np.arange(batch_size), selection_hard), norm_batch_size, replace=False)
    selection = np.append(selection_hard, selection_norm)

    triplet_training_batch = [random_anchor_batch[selection],
                              random_positive_batch[selection], random_negative_batch[selection]]
    batch_labels = random_labels[selection]

    return triplet_training_batch, batch_labels


def get_hard_pair_batch(args: dict, user_to_traces_map: dict, model: tf.keras.Model):
    """
    Generates a hard pair training batch.

    Parameters:
        args (dict): dictionary of arguments
        user_to_traces_map (dict): dictionary that holds for each user (key) all associated traces (value)
        model (tf.keras.Model): neural network model to perform a single forward pass 

    Returns:
        pair_training_batch (list): list of numpy arrays
        batch_labels (array): array of labels 
    """
    batch_size = args.batch_size * 2
    hard_batch_size = int(batch_size / 2)

    random_batch, random_labels = get_random_pair_batch(
        args, user_to_traces_map, batch_size)
    random_batch_x1, random_batch_x2 = random_batch

    # Perform the forward pass
    output = model.predict(random_batch)
    dist = np.squeeze(output)
    # Select samples that show the highest distance
    selection = tf.argsort(dist, direction="DESCENDING")[:hard_batch_size]

    pair_training_batch = [
        random_batch_x1[selection], random_batch_x2[selection]]
    batch_labels = random_labels[selection]

    return pair_training_batch, batch_labels


def get_semi_hard_pair_batch(args: dict, user_to_traces_map: dict, model: tf.keras.Model):
    """
    Generates a semi-hard pair training batch.

    Parameters:
        args (dict): dictionary of arguments
        user_to_traces_map (dict): dictionary that holds for each user (key) all associated traces (value)
        model (tf.keras.Model): neural network model to perform a single forward pass 

    Returns:
        pair_training_batch (list): list of numpy arrays
        batch_labels (array): array of labels 
    """

    batch_size = args.batch_size * 2
    hard_batch_size = int(batch_size / 4)
    norm_batch_size = int(batch_size / 4)

    random_batch, random_labels = get_random_pair_batch(
        args, user_to_traces_map, batch_size)
    random_batch_x1, random_batch_x2 = random_batch

    # Perform the forward pass
    output = model.predict(random_batch)
    dist = np.squeeze(output)
    # Select samples that show the highest distance
    selection_hard = tf.argsort(dist, direction="DESCENDING")[:hard_batch_size]
    # Select other samples randomly
    selection_norm = np.random.choice(np.delete(
        np.arange(batch_size), selection_hard), norm_batch_size, replace=False)
    selection = np.append(selection_hard, selection_norm)

    pair_training_batch = [
        random_batch_x1[selection], random_batch_x2[selection]]
    batch_labels = random_labels[selection]

    return pair_training_batch, batch_labels


def get_random_pair_batch(args: dict, user_to_traces_map: dict, batch_size=None):
    """
    Generates a random pair training batch.

    Parameters:
        args (dict): dictionary of arguments
        user_to_traces_map (dict): dictionary that holds for each user (key) all associated traces (value)
        batch_size (int): size of the training batch

    Returns:
        pair_training_batch (list): list of numpy arrays
        batch_labels (array): array of labels 
    """
    batch_x1 = []
    batch_x2 = []
    batch_labels = []

    if batch_size is None:
        batch_size = args.batch_size

    # Select 50% positive pairs
    for i in range(int(batch_size / 2)):
        random_user = random.sample(user_to_traces_map.keys(), 1)[0]
        traces_list = user_to_traces_map[random_user]
        sampled_traces = random.sample(traces_list, 2)
        trace_1 = sampled_traces[0]
        trace_2 = sampled_traces[1]
        batch_x1.append(trace_1)
        batch_x2.append(trace_2)
        batch_labels.append(1)

    # Select 50% negative pairs
    for i in range(int(batch_size / 2)):
        random_users = random.sample(user_to_traces_map.keys(), 2)
        traces_list_1 = user_to_traces_map[random_users[0]]
        traces_list_2 = user_to_traces_map[random_users[1]]
        trace_1 = random.sample(traces_list_1, 1)[0]
        trace_2 = random.sample(traces_list_2, 1)[0]
        batch_x1.append(trace_1)
        batch_x2.append(trace_2)
        batch_labels.append(0)

    # Shuffle all selected pairs of traces
    collection = list(zip(batch_x1, batch_x2, batch_labels))
    random.shuffle(collection)
    batch_x1, batch_x2, batch_labels = zip(*collection)

    pair_training_batch = [np.asarray(
        batch_x1, dtype='float32'), np.asarray(batch_x2, dtype='float32')]
    batch_labels = np.asarray(batch_labels)

    return pair_training_batch, batch_labels


def get_test_pair_batch(args: dict, user_to_traces_map: dict, user_to_test_idx_map: dict, target_trace: list, target_user: int, p: int):
    """
    Generates a pair testing batch.

    Parameters:
        args (dict): dictionary of arguments
        user_to_traces_map (dict): dictionary that holds for each user (key) all associated traces (value)
        user_to_test_idx_map (dict): dictionary that holds for each user (key) a randomly selected trace (value)
        target_trace (list): selected target trace for the linkage attack
        target_user (int): user identifer that the target trace belongs to
        p (int): index to ensure not not more traces are selected than the batch size

    Returns:
        pair_training_batch (list): list of numpy arrays
        bat
    """
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

    pair_testing_batch = [np.asarray(
        batch_x1, dtype='float32'), np.asarray(batch_x2, dtype='float32')]
    batch_labels = np.asarray(batch_labels)

    return pair_testing_batch, batch_labels


def get_test_triplet_batch(args: dict, user_to_traces_map: dict, user_to_test_idx_map: dict, target_trace: list, target_user: int, p: int):
    """
    Generates a triplet testing batch.

    Parameters:
        args (dict): dictionary of arguments
        user_to_traces_map (dict): dictionary that holds for each user (key) all associated traces (value)
        user_to_test_idx_map (dict): dictionary that holds for each user (key) a randomly selected trace (value)
        target_trace (list): selected target trace for the linkage attack
        target_user (int): user identifer that the target trace belongs to
        p (int): index to ensure not not more traces are selected than the batch size

    Returns:
        pair_training_batch (list): list of numpy arrays
        bat
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

        # Make sure that the negative trace is truely a negative trace, i.e., negative user != target user
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

    triplet_testing_batch = [np.asarray(batch_anchor, dtype='float32'), np.asarray(
        batch_positive, dtype='float32'), np.asarray(batch_negative, dtype='float32')]
    batch_labels = np.asarray(batch_labels)

    return triplet_testing_batch, batch_labels
