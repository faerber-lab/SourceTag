def restore_qwen_text_architecture(model):
    """Restore missing metadata after Unsloth extracts Qwen's text config,
    since the config can contain None.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    config = base.config
    if getattr(config, "model_type", None) != "qwen3_5_text":
        return False
    if getattr(config, "architectures", None):
        return False
    if not any(cls.__name__ == "Qwen3_5ForCausalLM" for cls in type(base).__mro__):
        raise TypeError(f"Expected Qwen3_5ForCausalLM for text-only Qwen, got {type(base).__name__}")
    config.architectures = ["Qwen3_5ForCausalLM"]
    print("Restored Qwen text-only config.architectures=['Qwen3_5ForCausalLM'] for Unsloth generation.")
    return True
