from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InferArguments:
    """
    Arguments for time series model configuration.
    """

    device: str = field(default="cuda:0")
    model_path: str = field(
        default="weights/model.safetensors", metadata={"help": ("Model weight path.")}
    )
    input_size: int = field(
        default=512, metadata={"help": ("Input sequence length of Transformer encoder")}
    )
    horizon: int = field(default=128, metadata={"help": ("The prediction horizon.")})
    hidden_size: int = field(
        default=768, metadata={"help": ("The dimension of each time point.")}
    )
    patch_size: int = field(
        default=32, metadata={"help": ("The patch size for target and exog variables.")}
    )
    stride: int = field(
        default=32,
        metadata={
            "help": (
                "The stride size of patch, when it is greater than the patch size, "
                "it means there is no overlap; otherwise, there is an overlap. "
            )
        },
    )
    n_head: int = field(default=12)
    encoder_layers: int = field(default=8)
    dropout: float = field(default=0.1)
    activation: str = field(
        default="GELU",
        metadata={
            "help": (
                "The dimension of the conv1d projection layer in each attention block."
            ),
            "choices": [
                "ReLU",
                "Softplus",
                "Tanh",
                "SELU",
                "LeakyReLU",
                "PReLU",
                "Sigmoid",
                "GELU",
            ],
        },
    )
    max_num_exogs: int = field(
        default=16,
        metadata={
            "help": (
                "The max number of exog variables. Above the num will be truncated, else will be padded."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    ffn_hidden_size: int = field(
        default=3072,
        metadata={"help": ("The hidden layer size of the feed-forward network")},
    )
