from src.util.filereader import load_abstracts, load_path_file, write_path_file, file_exists
from src.util.key_generator import generate_keys
from src.util.watermark import watermark, init_watermarker, verify_watermarks
import json 
import os 
from pathlib import Path 
import sys
sys.stdout.reconfigure(line_buffering=True)

experiment_name = "4_kappa_ablation"
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "sub_experiment_name",
    nargs="?",              # makes it optional
    default=""
)
args = parser.parse_args()
print(f"Using: {args.sub_experiment_name}")
sub_experiment_name = args.sub_experiment_name
if sub_experiment_name == "":
    sub_experiment_names = ["two", "four", "six", "eight"]
else:
    sub_experiment_names = [sub_experiment_name]

default_config = load_path_file(["configs", "watermark", experiment_name], "default_config.json")

n_samples = default_config["n_samples"]

T_os = load_abstracts(n_samples)
print(T_os)

# generate keys
for sub_experiment_name in sub_experiment_names:
    config = load_path_file(["configs", "watermark", experiment_name, sub_experiment_name], "config.json")
    config = config | default_config # combine configs
    print(config)
    id_gen_type = config["id_gen"]
    kp_gen_type = config["k_p_gen"]
    config_path = ["configs", "watermark", experiment_name, sub_experiment_name]
    if file_exists(config_path, "keys.json"):
        keys = load_path_file(config_path, "keys.json")
    else:
        keys = generate_keys(n_samples, id_type=id_gen_type, kp_type=kp_gen_type)
        write_path_file(config_path, "keys.json", keys)
    n_samples = len(keys["ids"])
    if len(keys["k_ps"]) != n_samples:
        raise ValueError("Misaligned saved watermark keys")
    T_os = load_abstracts(n_samples)
    ids = keys["ids"]
    k_ps = keys["k_ps"]

    tokenizer, model, watermarker = init_watermarker(config)
    # watermark 
    experiment_path = ["results", "_ablations", "watermark", experiment_name, sub_experiment_name]
    T_ws, sts_scores = watermark(T_os, ids, k_ps, config, experiment_path, return_sts_scores=True)
    # verify
    
    verify_watermarks(T_ws, [ids[0]], k_ps, watermarker, experiment_path)
