import argparse
import random
import tensorflow as tf
import logging

from utils import load_data
from utils import gen_target_user_to_target_trace_map
from model import SiameseTripletLoss
from model import TripletLoss
from sample import get_triplet_batch_hard

def main():

    parser = argparse.ArgumentParser(description="Perform a Linkage Attack.")
    parser.add_argument("--max_trace_len", type=int, default=500, help="Specifies the maximum length of a trace.")
    parser.add_argument("--min_trace_len", type=int, default=10, help="Specifies the minimum length of a trace.")
    parser.add_argument("--min_num_traces_per_user", type=int, default=4, help="Specifies the minimum number of traces per user.")
    parser.add_argument("--max_trace_duration", type=float, default=86400.0, help="Specifies the maximum duration of a trace.")
    parser.add_argument("--max_delay", type=float, default=1800.0, help="Specifies the maximum delay between two consecutive samples.")
    parser.add_argument("--epochs", type=int, default=20, help="Specifies the number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=128, help="Specifies the batch size to train the model.")
    parser.add_argument("--seed", type=int, default=0, help="Specifies the seed.")
    parser.add_argument("--path", type=str, default="", help="Specifies the absolute path to the dataset.")
    parser.add_argument("--gpu", type=str, default="/device:GPU:1", help="Specifies the gpu to train on. Default: /device:GPU:1")
    parser.add_argument("--dropout", type=float, default=0.4, help="Specifies the dropout ratio.")
    parser.add_argument("--sampling_strategy", type=str, default="random", choices=["random", "hard"], help="Specifies the sampling strategy.")
    parser.add_argument("--approach", type=str, default="cl", choices=["tl", "cl"], help="Specifies the approach: triplet Loss or contrastive Loss.")
    parser.add_argument("--sample_size", type=int, default=400, help="Specifies the sample size or number of linkage attacks to perform.")
    parser.add_argument("--margin", type=float, default=1.0, help="Specifies the margin.")
    parser.add_argument("--latent_size", type=int, default=128, help="Specifies the margin.")
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(message)s",level=logging.INFO)
    logging.info(args)

    # Set seed for reproducability
    random.seed(args.seed)
    tf.random.set_seed(args.seed)

    logging.info("Loading data...")
    user_to_traces_map = load_data(args)
    users = list(user_to_traces_map.keys())
    num_users = len(users)

    train_user_to_traces_map = {k: v[:int(len(v) / 2)] for k, v in user_to_traces_map.items()}
    train_num_samples = sum([len(v) for k, v in train_user_to_traces_map.items()])

    target_user_list = random.sample(users, args.sample_size)   
    target_user_to_target_trace_map = gen_target_user_to_target_trace_map(user_to_traces_map, target_user_list)

    logging.info("Building network...")
    model = SiameseTripletLoss(args)
    triplet_loss = TripletLoss(args)
    model.compile(optimizer="adam", loss=triplet_loss)

    steps_per_epoch = train_num_samples // args.batch_size

    logging.info("Beginning to train the network...")
    for epoch in range(args.epochs):
        total_loss = 0
        for step in range(steps_per_epoch):
            batch_x, batch_labels = get_triplet_batch_hard(args, train_user_to_traces_map, model)
            loss = model.train_on_batch(batch_x, batch_labels)
            total_loss += loss

        logging.info("Epoch [%i|%i] -- Avg Loss per Epoch: %.4f", epoch + 1, args.epochs, total_loss / steps_per_epoch)

        if epoch % 2 == 0:
            logging.info("Validating the network...")  
            preds = []
            for i in range(args.sample_size):
                target_user = target_user_list[i]
                target_trace = target_user_to_target_trace_map[target_user]
                for p in range(0, num_users, args.batch_size):
                    batch_x, _ = get_test_batch(user_to_traces_map, target_trace, target_user, p)
                    batch_preds = model.predict_on_batch(batch_x)
                    batch_preds_anchor, batch_preds_positive, batch_preds_negative = batch_preds[:,:args.latent_size], batch_preds[:,args.latent_size:2*args.latent_size], batch_preds[:,2*args.latent_size:]
                    batch_preds_dist = tf.reduce_mean(tf.square(batch_preds_anchor - batch_preds_positive), axis=1)
                    preds.append(np.squeeze(batch_preds_dist))  

            j = 0 
            in_top_1 = 0
            in_top_10 = 0
            in_top_10_percent = 0
            preds = np.concatenate(preds)
            for i in range(0, preds.shape[0], num_users):
                preds_attack = preds[i:i+num_users]
                sorted_idx = np.argsort(preds_attack)
                sorted_users = [users[i] for i in sorted_idx]
                target_user = target_user_list[j]
                if target_user == sorted_users[0]:
                    in_top_1 += 1
                if target_user in sorted_users[:10]:
                    in_top_10 += 1
                split = int(0.1 * len(users))
                if target_user in sorted_users[:split]:
                    in_top_10_percent += 1
                j += 1

            logging.info("Top 1: %.4f -- Top 10: %.4f -- Top 10%: %.4f", in_top_1 / args.sample_size, in_top_10 / args.sample_size, in_top_10_percent / args.sample_size)

    



if __name__ == "__main__":
    main()