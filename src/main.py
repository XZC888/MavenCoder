import os
import argparse
from mavencoder import run
import datetime
from log_helper import setup_logger
from dataset_processor import convert_format

timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        help="The output data directory", default="./output")
    parser.add_argument("--output_path", type=str,
                        help="User specified output data path if provided. Should be a JSONL file.")
    parser.add_argument("--log_dir", type=str,
                        help="The log directory", default=f"./log/{timestamp}")
    parser.add_argument("--test_dir", type=str,
                        help="The final processed test file directory", default=f"./test")
    
    parser.add_argument("--dataset_type", type=str,
                        help="dataset type", required=True) # choices=["lcb", "code_contests", "humanevalplus", "mbppplus"]
    parser.add_argument("--model", type=str, 
                        help="model names", required=True)

    parser.add_argument("--strategy", type=str,
                        help="difficulty assessment strategy", default="mean_prob", choices=["mean_prob", "entropy", "prompt"])
    parser.add_argument("--theta_1", type=float,
                        help="confidence threshold_1 (Low)", default=0.15)
    parser.add_argument("--theta_2", type=float,
                        help="confidence threshold_2 (High)", default=0.45)
    parser.add_argument("--r_global", type=int,
                        help="The maximum number of global iterations", default=3)
    parser.add_argument("--r_debug", type=int,
                        help="The maximum number of debug iterations", default=3)
    parser.add_argument("--r_valid", type=int,
                        help="The maximum number of verification iterations", default=1)
    
    parser.add_argument("--verbose", type=bool,
                        help="If True, displays detailed logs in the console.", default=False, choices=[True, False])
    parser.add_argument("--max_workers", type=int, 
                        help="max concurrent request workers", default=3)
    parser.add_argument("--key", type=str, help="API key", required=True)
    parser.add_argument("--url", type=str, help="URL")
    parser.add_argument("--embedding_key", type=str, help="Embedding API key (if different from main API key)", default="")
    parser.add_argument("--embedding_url", type=str, help="Embedding API URL (if different from main API URL)", default="")
    parser.add_argument("--embedding_model", type=str, help="Embedding model name", default="text-embedding-3-large")
    args = parser.parse_args()
    return args


def main(args):
    if not args.output_path:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"{args.dataset_type}_{args.model}_global={args.r_global}_debug={args.r_debug}_valid={args.r_valid}_strategy={args.strategy}.jsonl")
    else:
        output_path = args.output_path
    
    if not output_path.endswith('.jsonl'):
        raise ValueError("Output file should be JSONL format.")
    
    os.makedirs(args.log_dir, exist_ok=True)

    global_logger = setup_logger(os.path.join(args.log_dir, "global.log"), args.verbose, "a")

    global_logger.info(f"""\nStart running with the following parameters:
args: {args}
Logs will be saved in `{args.log_dir}`
""")

    run(
        dataset_type=args.dataset_type,
        model_name=args.model,
        strategy=args.strategy,
        theta_1=args.theta_1,
        theta_2=args.theta_2,
        r_global=args.r_global,
        r_debug=args.r_debug,
        r_valid=args.r_valid,
        output_path=output_path,
        log_dir = args.log_dir,
        key=args.key,
        url=args.url,
        embedding_key=args.embedding_key,
        embedding_url=args.embedding_url,
        embedding_model=args.embedding_model,
        verbose=args.verbose,
        max_workers=args.max_workers,
    )

    print(f"Done! Check out the output data in `{output_path}`")
    convert_format(output_path, args.dataset_type, args.test_dir)
    print(f"Processed test data in `{args.test_dir}`")


if __name__ == "__main__":
    args = get_args()
    main(args)
