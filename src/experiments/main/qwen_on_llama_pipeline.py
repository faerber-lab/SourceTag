"""Trains Qwen3.5-9B on the Llama-watermarked corpus.
"""

from src.experiments.main.qwen_pipeline import main as _qwen_main


def main(argv=None):
    return _qwen_main("watermarked", argv, watermark_source="llama")


if __name__ == "__main__":
    main()
