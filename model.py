import tensorflow as tf


class SiameseTripletLoss(tf.keras.Model):
    """
    Siamese neural network trained with triplet loss, implemented with Tensorflow.
    """

    def __init__(self, args: dict):
        super(SiameseTripletLoss, self).__init__()
        self.args = args
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            self.args.max_trace_len, return_sequences=False))
        self.d1 = tf.keras.layers.Dense(units=256)
        self.d2 = tf.keras.layers.Dense(units=128)
        self.dropout = tf.keras.layers.Dropout(self.args.dropout)

    def call(self, x):
        x_anchor, x_pos, x_neg = x

        emb_anchor = self.lstm(x_anchor)
        emb_anchor = self.dropout(emb_anchor)
        emb_anchor = self.d1(emb_anchor)
        emb_anchor = self.d2(emb_anchor)

        emb_pos = self.lstm(x_pos)
        emb_pos = self.dropout(emb_pos)
        emb_pos = self.d1(emb_pos)
        emb_pos = self.d2(emb_pos)

        emb_neg = self.lstm(x_neg)
        emb_neg = self.dropout(emb_neg)
        emb_neg = self.d1(emb_neg)
        emb_neg = self.d2(emb_neg)

        output = tf.keras.layers.concatenate(
            [emb_anchor, emb_pos, emb_neg], axis=1)
        return output


class TripletLoss(tf.keras.losses.Loss):
    """
    Triplet loss implementation.
    """

    def __init__(self, args):
        super(TripletLoss, self).__init__()
        self.args = args

    def call(self, _, y_pred):
        anchor, pos, neg = y_pred[:, :self.args.latent_size], y_pred[:,
                                                                     self.args.latent_size:2*self.args.latent_size], y_pred[:, 2*self.args.latent_size:]
        pos_dist = tf.reduce_mean(tf.square(anchor - pos), axis=1)
        neg_dist = tf.reduce_mean(tf.square(anchor - neg), axis=1)
        return tf.maximum(pos_dist - neg_dist + self.args.margin, 0.0)


class SiameseContrastiveLoss(tf.keras.Model):
    """
    Siamese neural network trained with contrastive loss, implemented with Tensorflow.
    """

    def __init__(self, args):
        super(SiameseContrastiveLoss, self).__init__()
        self.args = args
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            self.args.max_trace_len, return_sequences=False, dropout=self.args.dropout))
        self.d1 = tf.keras.layers.Dense(units=256)
        self.d2 = tf.keras.layers.Dense(units=256)
        self.dist = tf.keras.layers.Lambda(self.distance)

    def call(self, x):
        x1, x2 = x

        x1 = self.lstm(x1)
        x1 = self.d1(x1)

        x2 = self.lstm(x2)
        x2 = self.d2(x2)

        x = self.dist([x1, x2])
        return x

    def distance(self, x):
        x1, x2 = x
        sum_square = tf.reduce_sum(tf.square(x1 - x2), axis=1, keepdims=True)
        return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))


class ContrastiveLoss(tf.keras.losses.Loss):
    """
    Contrastive loss implementation.
    """

    def __init__(self, args):
        super(ContrastiveLoss, self).__init__()
        self.args = args

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(self.args.margin - (y_pred), 0))
        return tf.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )
