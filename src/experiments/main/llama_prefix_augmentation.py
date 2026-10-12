"""M=3/5 Llama recitation experiment."""
from src.experiments.main.qwen_pipeline import main

if __name__ == '__main__':
    main(watermark_source='llama', training_source='llama')
