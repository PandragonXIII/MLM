import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cfg-path", default="./eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--image_num", default=100, type=int)
parser.add_argument("--input_res", default=224, type=int)
parser.add_argument("--alpha", default=1.0, type=float)
parser.add_argument("--epsilon", default=8, type=int)
parser.add_argument("--steps", default=8, type=int)
parser.add_argument("--output", default="temp", type=str)
parser.add_argument("--image_path", default="temp", type=str)
parser.add_argument("--tgt_text_path", default="temp.txt", type=str)
parser.add_argument("--question_path", default="temp.txt", type=str)
parser.add_argument("--start_idx", default=0, type=int)
parser.add_argument("--num_query", default=10, type=int)
parser.add_argument("--sigma", default=8, type=float)

# parser.add_argument("--wandb", action="store_true")
# parser.add_argument("--wandb_project_name", type=str, default='temp_proj')
# parser.add_argument("--wandb_run_name", type=str, default='temp_run')

args = parser.parse_args()
a = args.options
print(a)