# Repository Structure
`configs/` contains configuration files for the training.

`data` contains the seeded, unwatermarked texts in `seeded_dataset.jsonl`, the used watermarking `keys.json`, as well as configuration files.

`data/t_ws` contains generation parameters and the watermarked texts (T_ws) for qwen and llama. The evaluation prompts are in `data/prompts`.

`training_metadata/{trainmodel-on-watermarkmodel}/{task}/{corpus_size}/{batch_size}/` contains the loss trajectories for each run. 

`src/` contains the source code, mainly the data generation pipeline in `src/data_creation/`, the main experiment pipelines in `src/experiments` as well as utilities in `src/util`.

# Create environments
Use Python 3.12 and create both environments from the repository root.
Due to incompatible transformer version requirements between Unsloth and Waterfall,
two separate environments are needed.
Create the watermarking environment:

```bash
python -m venv .venv-watermark
.venv-watermark/bin/python -m pip install --upgrade pip
.venv-watermark/bin/python -m pip install -r req-watermark.txt
```

Create the training/evaluation environment:

```bash
python -m venv .venv-experiment
.venv-experiment/bin/python -m pip install --upgrade pip
.venv-experiment/bin/python -m pip install -r req-experiment.lock.txt
```

## Which environment to use
Use `.venv-watermark` for:

- `src.data_creation.create_t_ws` 
- `src.data_creation.create_prompt_dataset`
- `src.experiments.ablations.qwen_kappa_ablation`
- all `submit_qwen_*ablation.sh` and `submit_qwen_*corpus.sh` helpers

Use `.venv-experiment` for:

- all main experiments with fine-tuning and evaluation
- closed/open-keyspace evaluation and additional similarity metrics
