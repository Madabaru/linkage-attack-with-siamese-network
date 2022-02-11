import tensorflow as tf


def get_triplet_random_batch(batch_size, user_to_traces_map: dict):
    """ Generates a random triplet training batch. """
    batch_anchor = []
    batch_positive = []
    batch_negative = []
    batch_labels = []

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


def get_triplet_batch_hard(args: dict, user_to_traces_map: dict, model: tf.keras.Model):
    """ Generates a hard triplet training batch. """
    batch_size = args.batch_size * 2

    hard_batch_size = int(batch_size / 4)
    norm_batch_size = int(batch_size / 4)
    
    random_batch, random_labels = get_triplet_random_batch(batch_size, user_to_traces_map)
    random_anchor_batch, random_positive_batch, random_negative_batch = random_batch

    output = model.predict(random_batch)
    anchor, positive, negative = output[:, :args.latent_size], output[: ,args.latent_size:2*args.latent_size], output[:, 2*args.latent_size:]
    dist = tf.reduce_mean(tf.square(anchor - positive), axis=1) - tf.reduce_mean(tf.square(anchor - negative), axis=1)
    selection_hard = tf.argsort(dist, direction="DESCENDING")[:hard_batch_size]

    # Select other random samples
    selection_norm = np.random.choice(np.delete(np.arange(batch_size),selection_hard), norm_batch_size,replace=False)
    selection = np.append(selection_hard, selection_norm)
    
    return [random_anchor_batch[selection], random_positive_batch[selection], random_negative_batch[selection]], random_labels[selection]