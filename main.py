import argparse
import random
import tensorflow as tf
import logging
import numpy as np

from utils import load_data, gen_user_to_sample_idx_map, gen_user_to_test_idx_map, gen_user_to_target_idx_map
from model import SiameseContrastiveLoss, SiameseTripletLoss, ContrastiveLoss, TripletLoss
from sample import get_hard_triplet_batch, get_semi_hard_triplet_batch, get_random_triplet_batch, get_test_triplet_batch
from sample import get_random_pair_batch, get_test_pair_batch, get_semi_hard_pair_batch, get_hard_pair_batch


def main():

    parser = argparse.ArgumentParser(
        description="Perform a linkage attacks using a Siamese neural network.")
    parser.add_argument("--max_trace_len", type=int, default=500,
                        help="Specifies the maximum length of a trace.")
    parser.add_argument("--min_trace_len", type=int, default=10,
                        help="Specifies the minimum length of a trace.")
    parser.add_argument("--min_num_traces_per_user", type=int, default=4,
                        help="Specifies the minimum number of traces per user.")
    parser.add_argument("--max_trace_duration", type=float, default=86400.0,
                        help="Specifies the maximum duration of a trace.")
    parser.add_argument("--max_delay", type=float, default=1800.0,
                        help="Specifies the maximum delay between two consecutive samples.")
    parser.add_argument("--epochs", type=int, default=400,
                        help="Specifies the number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Specifies the batch size to train the model.")
    parser.add_argument("--seed", type=int, default=1,
                        help="Specifies the seed.")
    parser.add_argument("--path", type=str, default="/home/john/data/mobility/driving_sampled_8k.csv",
                        help="Specifies the absolute path to the dataset.")
    parser.add_argument("--gpu", type=str, default="/device:GPU:0",
                        help="Specifies the gpu to train on. Default: /device:GPU:0")
    parser.add_argument("--dropout", type=float, default=0.4,
                        help="Specifies the dropout ratio.")
    parser.add_argument("--sampling_strategy", type=str, default="hard", choices=[
                        "random", "semi_hard", "hard"], help="Specifies the sampling strategy.")
    parser.add_argument("--approach", type=str, default="tl", choices=[
                        "tl", "cl"], help="Specifies the approach and loss function: triplet loss or contrastive Loss.")
    parser.add_argument("--sample_size", type=int, default=400,
                        help="Specifies the sample size or number of linkage attacks to perform.")
    parser.add_argument("--num_target_traces", type=int, default=1,
                        help="Specifies the number of target traces per user.")
    parser.add_argument("--margin", type=float, default=1.0,
                        help="Specifies the margin.")
    parser.add_argument("--latent_size", type=int,
                        default=128, help="Specifies the margin.")
    parser.add_argument("--k", type=int, default=0,
                        help="Specifies the number of observed traces per user.")
    parser.add_argument("--fields", type=list, nargs='+', help="Specifies the fields to consider on the mobility dataset.",
                        default=["state", "street", "postcode", "speed"])
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    logging.info(args)

    with tf.device(args.gpu):

        # Set seed for reproducability
        random.seed(args.seed)
        tf.random.set_seed(args.seed)

        logging.info("Loading data...")
        user_to_traces_map = load_data(args)
        users = list(user_to_traces_map.keys())
        num_users = len(users)

        logging.info("Sampling data...")
        user_to_sample_idx_map = gen_user_to_sample_idx_map(
            user_to_traces_map, args.k)

        train_user_to_traces_map = {user: [e for i, e in enumerate(user_to_traces_map.get(
            user)) if i in traces_list] for user, traces_list in user_to_sample_idx_map.items()}
        train_num_samples = sum([len(v)
                                for k, v in train_user_to_traces_map.items()])

        user_to_target_idx_map = gen_user_to_target_idx_map(
            args, user_to_traces_map)
        target_user_list = list(user_to_target_idx_map.keys())

        user_to_test_idx_map = gen_user_to_test_idx_map(user_to_sample_idx_map)

        logging.info("Building network...")
        if args.approach == "tl":
            model = SiameseTripletLoss(args)
            triplet_loss = TripletLoss(args)
            model.compile(optimizer="adam", loss=triplet_loss)
        else:
            model = SiameseContrastiveLoss(args)
            contrastive_loss = ContrastiveLoss(args)
            model.compile(optimizer="adam", loss=contrastive_loss)

        logging.info("Beginning to train the network...")
        steps_per_epoch = train_num_samples // args.batch_size
        for epoch in range(args.epochs):
            total_loss = 0
            for _ in range(steps_per_epoch):
                if args.approach == "tl":
                    if args.sampling_strategy == "hard":
                        batch_x, batch_labels = get_hard_triplet_batch(
                            args, train_user_to_traces_map, model)
                    elif args.sampling_strategy == "semi_hard":
                        batch_x, batch_labels = get_semi_hard_triplet_batch(
                            args, train_user_to_traces_map, model)
                    else:
                        batch_x, batch_labels = get_random_triplet_batch(
                            args, train_user_to_traces_map)
                else:
                    if args.sampling_strategy == "hard":
                        batch_x, batch_labels = get_hard_pair_batch(
                            args, train_user_to_traces_map, model)
                    elif args.sampling_strategy == "semi_hard":
                        batch_x, batch_labels = get_semi_hard_pair_batch(
                            args, train_user_to_traces_map, model)
                    else:
                        batch_x, batch_labels = get_random_pair_batch(
                            args, train_user_to_traces_map)

                loss = model.train_on_batch(batch_x, batch_labels)
                total_loss += loss

            logging.info("Epoch [%i|%i] -- Avg Loss per Epoch: %.4f" %
                         (epoch + 1, args.epochs, total_loss / steps_per_epoch))

            if epoch % 10 == 0:
                logging.info("Validating the network...")
                preds = []
                for i in range(args.sample_size):
                    target_user = target_user_list[i]
                    target_trace_idx_list = user_to_target_idx_map.get(
                        target_user)
                    for _, target_trace_idx in enumerate(target_trace_idx_list):
                        target_trace = user_to_traces_map.get(
                            target_user)[target_trace_idx]
                        for p in range(0, num_users, args.batch_size):
                            if args.approach == "tl":
                                batch_x, _ = get_test_triplet_batch(
                                    args, user_to_traces_map, user_to_test_idx_map, target_trace, target_user, p)
                                batch_preds = model.predict_on_batch(batch_x)
                                batch_preds_anchor, batch_preds_positive, _ = batch_preds[:, :args.latent_size], batch_preds[
                                    :, args.latent_size:2*args.latent_size], batch_preds[:, 2*args.latent_size:]
                                output = tf.reduce_mean(
                                    tf.square(batch_preds_anchor - batch_preds_positive), axis=1)
                                preds.extend(list(output.numpy()))
                            else:
                                batch_x, _ = get_test_pair_batch(
                                    args, user_to_traces_map, user_to_test_idx_map, target_trace, target_user, p)
                                output = model.predict_on_batch(batch_x)
                                preds.extend(list(np.squeeze(output)))

                # Calculate top 1, top 10 and top 10% scores
                j = 0
                in_top_1_list = []
                in_top_10_list = []
                in_top_10_percent_list = []
                for i in range(0, len(preds), num_users):
                    preds_attack = preds[i:i+num_users]
                    if args.approach == "tl":
                        sorted_idx = np.argsort(preds_attack)
                    else:
                        sorted_idx = np.flip(np.argsort(preds_attack))
                    sorted_users = [users[i] for i in sorted_idx]
                    target_user = target_user_list[j]
                    if target_user == sorted_users[0]:
                        in_top_1_list.append(1)
                    else:
                        in_top_1_list.append(0)
                    if target_user in sorted_users[:10]:
                        in_top_10_list.append(1)
                    else:
                        in_top_10_list.append(0)
                    split = int(0.1 * len(users))
                    if target_user in sorted_users[:split]:
                        in_top_10_percent_list.append(1)
                    else:
                        in_top_10_percent_list.append(0)
                    j += 1
                logging.info("Top 1: %.4f Top 1 (Std): %.4f - Top 10: %.4f Top 10 (Std): %.4f - Top 10 Percent: %.4f Top 10 Percent (Std): %.4f" % (np.mean(
                    in_top_1_list), np.std(in_top_1_list), np.mean(in_top_10_list), np.std(in_top_10_list), np.mean(in_top_10_percent_list), np.std(in_top_10_percent_list)))


if __name__ == "__main__":
    main()
