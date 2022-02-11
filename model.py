import tensorflow as tf

class SiameseTripletLoss(tf.keras.Model):
  def __init__(self, args: dict):
    super(SiameseTripletLoss, self).__init__()
    self.args = args
    self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.args.max_trace_len, return_sequences = False))
    self.d1 = tf.keras.layers.Dense(units=256) 
    self.d2 = tf.keras.layers.Dense(units=128)
    self.dropout = tf.keras.layers.Dropout(self.args.dropout)
    
  def call(self, x):
    input_anchor, input_positive, input_negative = x
    
    embedding_anchor = self.lstm(input_anchor)
    embedding_anchor = self.dropout(embedding_anchor)
    embedding_anchor = self.d1(embedding_anchor)
    embedding_anchor = self.d2(embedding_anchor)

    embedding_positive = self.lstm(input_positive)
    embedding_positive = self.dropout(embedding_positive)
    embedding_positive = self.d1(embedding_positive)
    embedding_positive = self.d2(embedding_positive)

    embedding_negative = self.lstm(input_negative)
    embedding_negative = self.dropout(embedding_negative)
    embedding_negative = self.d1(embedding_negative)
    embedding_negative = self.d2(embedding_negative)

    output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)
    return output


class TripletLoss(tf.keras.losses.Loss):

    def __init__(self, args):
        super(TripletLoss, self).__init__()
        self.args = args

    def triplet_loss(self, y_true, y_pred):
        anchor, positive, negative = y_pred[:, :self.args.latent_size], y_pred[:, self.args.latent_size:2*self.args.latent_size], y_pred[:, 2*self.args.latent_size:]
        positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
        negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
        return tf.maximum(positive_dist - negative_dist + self.args.margin, 0.0)