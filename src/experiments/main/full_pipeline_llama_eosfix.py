"""Run Llama experiments with isolated EOS-fixed outputs.
Usage:
    python -m src.experiments.main.full_pipeline_llama_eosfix \
        [n_samples] [sample_type] [batch_size] [n_eval_samples] [--eval-only]
"""

import runpy
import sys


if "--eos-fix" not in sys.argv:
    sys.argv.append("--eos-fix")

runpy.run_module("src.experiments.main.full_pipeline", run_name="__main__")
