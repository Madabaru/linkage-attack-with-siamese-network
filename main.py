import argparse
import random
import tensorflow as tf
import logging

def main():

    parser = argparse.ArgumentParser(description="Perform a Linkage Attack.")
    parser.add_argument("--max_trace_len", type=int, default=500, help="Specifies the maximum length of a trace.")
    parser.add_argument("--min_trace_len", type=int, default=10, help="Specifies the minimum length of a trace.")
    parser.add_argument("--min_num_traces_per_client", type=int, default=4, help="Specifies the minimum number of traces per client.")
    parser.add_argument("--max_trace_duration", type=float, default=86400.0, help="Specifies the maximum duration of a trace.")
    parser.add_argument("--max_delay", type=float, default=1800.0, help="Specifies the maximum delay between two consecutive samples.")
    parser.add_argument("--epochs", type=int, default=20, help="Specifies the number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=128, help="Specifies the batch size to train the model.")
    parser.add_argument("--seed", type=int, default=0, help="Specifies the seed.")
    parser.add_argument("--path", type=str, default=0, help="Specifies the absolute path to the dataset.")
    parser.add_argument("--gpu", type=str, default="/device:GPU:1", help="Specifies the gpu to train on. Default: /device:GPU:1")
    parser.add_argument("--dropout", type=float, default=0.4, help="Specifies the dropout ratio.")
    parser.add_argument("--sampling_strategy", type=str, default="random", choices=["random", "hard"], help="Specifies the sampling strategy.")
    parser.add_argument("--approach", type=str, default="cl", choices=["tl", "cl"], help="Specifies the approach: triplet Loss or contrastive Loss.")
    parser.add_argument("--sample_size", type=int, default=400, help="Specifies the sample size or number of linkage attacks to perform.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(args)

    # Set seed for reproducability
    random.seed(args.seed)
    tf.random.set_seed(args.seed)

    



if __name__ == "__main__":
    main()