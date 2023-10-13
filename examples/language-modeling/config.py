from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    BertConfig,
    BertForMaskedLM,
)

def load_config_and_model(model_name: str):
    if model_name.startswith("gpt3"):
        return load_gpt3(model_name)
    if model_name.startswith("bert"):
        return load_bert(model_name)
    raise NotImplementedError(f"Unknown model '{model_name}'")


def load_gpt3(model_name: str):
    config = GPT2Config.from_pretrained("gpt2")
    config.use_cache = False
    config.return_dict = False

    if model_name == "gpt3-small":
        config.n_embd = 768
        config.n_layer = 12
        config.n_head = 12
    elif model_name == "gpt3-medium":
        config.n_embd = 1024
        config.n_layer = 24
        config.n_head = 16
    elif model_name == "gpt3-large":
        config.n_embd = 1536
        config.n_layer = 24
        config.n_head = 16
    elif model_name == "gpt3-xl":
        config.n_embd = 2048
        config.n_layer = 24
        config.n_head = 16
    elif model_name == "gpt3-2.7b":
        config.n_embd = 2560
        config.n_layer = 32
        config.n_head = 32
    elif model_name == "gpt3-6.7b":
        config.n_embd = 4096
        config.n_layer = 32
        config.n_head = 32
    elif model_name == "gpt3-13b":
        config.n_embd = 5120
        config.n_layer = 40
        config.n_head = 40
    elif model_name == "gpt3-175b":
        config.n_embd = 12288
        config.n_layer = 96
        config.n_head = 96
    else:
        raise NotImplementedError(
            f"'{model_name}' is not a supported GPT3 variant.",
        )

    model = GPT2LMHeadModel(config)
    print(f"Loaded '{model_name}' with {count_parameters(model)/1e6:.2f}M parameters")

    return config, model


def load_bert(model_name: str):
    config = BertConfig.from_pretrained(model_name)
    config.return_dict = False

    model = BertForMaskedLM(config)
    print(f"Loaded '{model_name}' with {count_parameters(model)/1e6:.2f}M parameters")

    return config, model


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
