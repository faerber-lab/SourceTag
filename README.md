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

# Run Experiments
Main experiment pipelines can be run as follows, here for example for training llama on llama-watermarked texts, with 1,000 training texts, trained on ```abstracts_only``` (also accepts ```abstracts_and_titles``` and ```questions```), with batch-size 32 and 1,000 evaluation samples:
```
python -m src.experiments.main.full_pipeline \
    --profile llama \
    1000 abstracts_only 32 1000
```
In order to calculate metrics such as WBR, we have to create a cache of token bigrams for each text:
```
python -m src.experiments.main.compute_t_w_bigrams \
    --profile llama
```
Then, we can calculate all metrics for each available checkpoint:
```
python -m src.experiments.main.similarity_eval \
    --profile llama \
    --n_samples 1000 \
    --sample_type abstracts_only \
    --batch_size 32 \
    --metrics bm25 cosine bertscore bigram lcs \
    --target watermarked \
    --bm25_corpus original \
    --all-epochs
```
For open-keyspace experiments, start them as follows:
```
python -m src.experiments.main.open_keyspace_eval \
    --profile llama \
    1000 abstracts_only 32 1000
```
An oracle baseline for direct detection on watermarked texts (without fine-tuning):
```
python -m src.experiments.main.oracle_watermark_eval \
    --source llama \
    --n-samples 1000
```
