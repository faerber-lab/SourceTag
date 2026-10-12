"""
Small ablation for batch-64 training on Qwen, saved in ``_ablations/qwen_on_llama_batch64``.
"""

from src.experiments.main.qwen_pipeline import main as _qwen_main


def main(argv=None):
    return _qwen_main(
        "watermarked",
        argv,
        watermark_source="llama",
        experiment_variant="batch64",
    )


if __name__ == "__main__":
    main()
