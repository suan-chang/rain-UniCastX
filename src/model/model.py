import torch
import pandas as pd
from torch import nn
from typing import Optional, List
from safetensors.torch import load_file as safe_load_file

from src.utils.args import InferArguments
from src.utils.data_util import MODEL_PARAMS_DICT, data_preprocess, ACTIVATIONS, OFFSET_ALIAS_MAP


class RainTimeSeriesModel(nn.Module):
    def __init__(self, model_args: InferArguments):
        super(RainTimeSeriesModel, self).__init__()

        self.model_args = model_args
        self.torch_dtype = getattr(torch, self.model_args.torch_dtype)
        self.device = self.model_args.device
        self.ts_model = TimeSeriesModel(self.model_args)

        self.load_ts_model(self.model_args.model_path, self.model_args.device)

        self.target_scaler = TemporalNorm(
            scaler_type="standard",
            dim=1,
            num_features=1,
        )
        self.exog_scaler = TemporalNorm(
            scaler_type="standard",
            dim=1,
            num_features=1,
        )
        self.time_scaler = TemporalNorm(
            scaler_type="standard",
            dim=1,
            num_features=1,
        )

    def load_ts_model(self, model_path, device):
        state_dict = safe_load_file(model_path)
        self.load_state_dict(state_dict=state_dict)
        self.to(device=device, dtype=self.torch_dtype)
        self.eval()

    def forward(
        self,
        insample_y,
        insample_exog,
        insample_time,
        cross_mask=None,
        padding_mask=None,
    ):
        """
        Args:
            insample_y,       # [B, L, 1] L: input length
            insample_exog,    # [B, L, 16]
            insample_time,    # [B, L, 6]
            insample_mask,    # [B, L, 1]

        Returns:
            dec_out,          # [B, H] H: output horizon
        """
        mask = torch.ones_like(insample_y)
        x_enc = self.target_scaler.transform(insample_y, mask)
        exog_enc = (
            self.exog_scaler.transform(insample_exog, mask)
            if insample_exog is not None
            else None
        )
        time_enc = self.time_scaler.transform(insample_time, mask)
        x_input = torch.concat(
            [tensor for tensor in [x_enc, exog_enc, time_enc] if tensor is not None],
            dim=-1,
        )
        dec_out = self.ts_model(x_input, cross_mask, padding_mask)
        dec_out = self.target_scaler.inverse_transform(dec_out.unsqueeze(-1)).squeeze(
            -1
        )

        return dec_out

    def predict(
        self,
        data: pd.DataFrame,
        time_col: str,
        target_col: str,
        freq: str,
        horizon: int,
        exog_cols: Optional[List[str]] = None,
        level: Optional[float] = None,
    ):
        insample_time, insample_y, insample_exog = data_preprocess(
            data=data,
            time_col=time_col,
            target_col=target_col,
            exog_cols=exog_cols,
            level=level,
            freq=freq,
            horizon=horizon,
            min_input_size=self.ts_model.patch_size,
            max_exog_nums=self.ts_model.max_exog_nums,
            torch_type=self.torch_dtype,
            device=self.device,
        )

        return self._predict_impl(
            insample_y=insample_y,
            insample_exog=insample_exog,
            insample_time=insample_time,
            horizon=horizon,
            level=level,
            freq=freq,
        )

    def _predict_impl(
        self,
        insample_y,
        insample_exog,
        insample_time,
        horizon=None,
        stride=128,
        level=None,
        freq="H",
        corrected=False,
    ):
        """
        Args:
            insample_y: [B, L, 1]. When level is not None, L should >> input size + horizon
            insample_exog: [B, L, 16]
            insample_time: [B, L, 6]
            horizon (int): The length of prediction.
            stride (int): The stride to generate calibration set. Defaults to 1.
            level (float): For a given significance level (error rate), the goal of CP is to return a prediction region
            that is guaranteed to contain the true value with probability of at least (1 - level).
            corrected (bool): Weather to use Bonferroni corrected calibration scores. Defaults to False.

        Returns:
            prediction: [B, H] H: output horizon
            boundary: [B, 2, H]. When level is not None.
        """
        if level:
            critical_calibration_scores_list = []
            corrected_critical_calibration_scores_list = []
            calibration_insample_y = insample_y
            calibration_insample_exog = insample_exog
            calibration_insample_time = insample_time
            freq = OFFSET_ALIAS_MAP[freq]
            model_input_size = MODEL_PARAMS_DICT[freq]["input_size"]

            for i in range(len(insample_y)):
                critical_calibration_scores, corrected_critical_calibration_scores = (
                    self.calculate_scores(
                        calibration_insample_y[i],
                        calibration_insample_exog[i],
                        calibration_insample_time[i],
                        model_input_size,
                        horizon,
                        stride,
                        level,
                    )
                )
                critical_calibration_scores_list.append(critical_calibration_scores)
                corrected_critical_calibration_scores_list.append(
                    corrected_critical_calibration_scores
                )

            critical_calibration_scores = torch.stack(
                critical_calibration_scores_list, dim=0
            )
            corrected_critical_calibration_scores = torch.stack(
                corrected_critical_calibration_scores_list, dim=0
            )

            input_insample_y = insample_y[:, -model_input_size:]
            input_insample_exog = insample_exog[:, -model_input_size:]
            input_insample_time = insample_time[:, -model_input_size:]
            with torch.no_grad():
                prediction = self.forward(
                    input_insample_y, input_insample_exog, input_insample_time
                ).squeeze(-1)
                prediction = prediction[:, :horizon]

            if not corrected:
                lower = prediction - critical_calibration_scores
                upper = prediction + critical_calibration_scores
            else:
                lower = prediction - corrected_critical_calibration_scores
                upper = prediction + corrected_critical_calibration_scores

            return prediction.float().cpu().numpy(), torch.stack(
                [lower, upper], dim=1
            ).float().cpu().numpy()
        else:
            with torch.no_grad():
                prediction = self.forward(
                    insample_y, insample_exog, insample_time
                ).squeeze(-1)
                prediction = prediction[:, :horizon]
            return prediction.float().cpu().numpy()

    def calculate_scores(
        self,
        insample_y,
        insample_exog,
        insample_time,
        input_size=None,
        horizon=None,
        stride=128,
        level=0.1,
    ):
        """
        Args:
            insample_y: [N, L, 1]
            insample_exog: [N, L, 16]
            insample_time: [N, L, 6]
            input_size (int): Input size.
            horizon (int): The length of prediction.
            stride (int): The stride to generate calibration set. Defaults to 1.
            level (float): For a given significance level (error rate), the goal of CP is to return a prediction region
            that is guaranteed to contain the true value with probability of at least (1 - level).
            corrected (bool): Weather to use Bonferroni corrected calibration scores. Defaults to False.
        """

        def _create_windows(x, window_size, stride=1):
            return x.unfold(dimension=0, size=window_size, step=stride)

        def _nonconformity(output, target):
            """
            Measures the nonconformity between output and target time series.

            Returns:
                Average MAE loss for every step in the sequence.
            """
            return torch.nn.functional.l1_loss(output, target, reduction="none")

        def _get_critical_scores(calibration_scores, q):
            """
            Computes critical calibration scores from scores in the calibration set.

            Args:
                calibration_scores: calibration scores for each example in the
                    calibration set.
                q: target quantile for which to return the calibration score

            Returns:
                critical calibration scores for each target horizon
            """
            device = calibration_scores.device
            return torch.tensor(
                [
                    torch.quantile(position_calibration_scores, q=q)
                    for position_calibration_scores in calibration_scores
                ],
                device=device,
            )

        window_size = input_size + horizon
        insample_y_windows = _create_windows(insample_y, window_size, stride).permute(
            0, 2, 1
        )
        insample_exog_windows = _create_windows(
            insample_exog, window_size, stride
        ).permute(0, 2, 1)
        insample_time_windows = _create_windows(
            insample_time, window_size, stride
        ).permute(0, 2, 1)

        with torch.no_grad():
            prediction = self.forward(
                insample_y_windows[:, :input_size],
                insample_exog_windows[:, :input_size],
                insample_time_windows[:, :input_size],
            )

        ground_truth = insample_y_windows[:, -horizon:]
        prediction = prediction[:, -horizon:]

        calibration_scores = _nonconformity(prediction, ground_truth.squeeze(-1)).T
        n_calibration = calibration_scores.shape[-1]
        q = min((n_calibration + 1.0) * (1 - level) / n_calibration, 1)
        critical_calibration_scores = _get_critical_scores(
            calibration_scores=calibration_scores.float(), q=q
        )

        corrected_q = min(
            (n_calibration + 1.0) * (1 - level / horizon) / n_calibration, 1
        )
        corrected_critical_calibration_scores = _get_critical_scores(
            calibration_scores=calibration_scores.float(), q=corrected_q
        )

        return critical_calibration_scores, corrected_critical_calibration_scores


class TimeSeriesModel(nn.Module):
    def __init__(self, model_args: InferArguments):
        super(TimeSeriesModel, self).__init__()

        self.model_args = model_args
        self.torch_dtype = getattr(torch, self.model_args.torch_dtype)
        self.patch_size = self.model_args.patch_size
        self.stride = self.model_args.stride
        self.hidden_size = self.model_args.hidden_size
        self.input_size = self.model_args.input_size
        self.max_exog_nums = self.model_args.max_num_exogs
        self.horizon = self.model_args.horizon
        self.n_head = self.model_args.n_head
        self.ffn_hidden_size = self.model_args.ffn_hidden_size
        self.encoder_layer_num = self.model_args.encoder_layers
        self.dropout = self.model_args.dropout
        self.activation = self.model_args.activation

        self.encoder = TransEncoder(
            self.hidden_size,
            self.n_head,
            self.patch_size,
            self.ffn_hidden_size,
            self.encoder_layer_num,
            dropout=self.dropout,
            activation=self.activation,
        )

        self.output_layer = nn.Linear(self.hidden_size, self.horizon)

    def forward(self, x, cross_mask=None, padding_mask=None):
        B, T, C = x.shape

        x_enc = x.permute(0, 2, 1).contiguous()
        x_enc = x_enc.view(-1, T).unsqueeze(-1)
        x_out = self.encoder(x_enc, C, cross_mask, padding_mask)
        output = self.output_layer(x_out)
        return output


class MlpFFN(nn.Module):
    def __init__(self, hidden_size, mlp_hidden_size, activation, dropout):
        super(MlpFFN, self).__init__()
        self.mlp1 = nn.Linear(hidden_size, mlp_hidden_size)
        self.mlp2 = nn.Linear(mlp_hidden_size, hidden_size)
        assert activation in ACTIVATIONS, f"{activation} is not in {ACTIVATIONS}"
        self.activation = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.activation(self.mlp1(x))
        x = self.dropout(x)
        x = self.mlp2(x)
        return x


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@torch.jit.script
def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, :, : x.shape[-2], :]
    sin = sin[:, :, : x.shape[-2], :]

    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim_model: int, *_, **__):
        super().__init__()

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_model, 2).float() / dim_model))
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=1):
        seq_len = x.shape[seq_dimension]

        if (
            seq_len != self._seq_len_cached
            or self._cos_cached.device != x.device
            or self._cos_cached.dtype != x.dtype
        ):
            self._seq_len_cached = seq_len
            t = torch.arange(
                x.shape[seq_dimension], device=x.device, dtype=torch.float32
            )
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(x.dtype))
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, None, :, :].to(x.dtype)
            self._sin_cached = emb.sin()[None, None, :, :].to(x.dtype)

        return self._cos_cached, self._sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(
            k, seq_dimension=-2
        )

        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )


class TokenEmbedding(nn.Module):
    def __init__(self, hidden_size, patch_size, stride):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(
            in_channels=1,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=stride,
            padding=0,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)

        return x


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size, n_head, dropout=0.1):
        super(AttentionLayer, self).__init__()

        d_keys = hidden_size // n_head
        d_values = hidden_size // n_head

        self.query_projection = nn.Linear(hidden_size, d_keys * n_head)
        self.key_projection = nn.Linear(hidden_size, d_keys * n_head)
        self.value_projection = nn.Linear(hidden_size, d_values * n_head)
        self.out_projection = nn.Linear(d_values * n_head, hidden_size)
        self.n_head = n_head
        self.dropout = dropout
        self.rope = RotaryEmbedding(hidden_size)

    def forward(
        self, queries, keys, values, cross_mask=None, padding_mask=None, use_RoPE=False
    ):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_head
        attn_mask = None

        assert (
            cross_mask is None or padding_mask is None
        ), "cross mask and padding mask can not both be none."
        if cross_mask is not None:
            cross_mask = (
                cross_mask.unsqueeze(1)
                .unsqueeze(1)
                .expand(-1, H, -1, -1)
                .to(torch.bfloat16)
            )
            cross_mask = cross_mask.masked_fill(cross_mask == 0, -float("inf"))
            cross_mask = cross_mask.masked_fill(cross_mask == 1, 0)
            attn_mask = cross_mask

        if padding_mask is not None:
            padding_mask = torch.cat(
                [
                    padding_mask,
                    torch.ones(padding_mask.shape[0], 1, device=padding_mask.device),
                ],
                dim=1,
            )
            padding_mask = padding_mask.unsqueeze(2) * padding_mask.unsqueeze(1)
            padding_mask = (
                padding_mask.unsqueeze(1).expand(-1, H, -1, -1).to(torch.bfloat16)
            )

            padding_mask = padding_mask.masked_fill(padding_mask == 0, -1000)
            padding_mask = padding_mask.masked_fill(padding_mask == 1, 0)
            channel_nums = B // padding_mask.shape[0]
            attn_mask = (
                padding_mask.unsqueeze(1)
                .expand(-1, channel_nums, -1, -1, -1)
                .contiguous()
                .view(-1, *padding_mask.shape[1:])
            )

        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)

        if use_RoPE:
            queries, keys = self.rope(queries, keys)

        queries = queries.view(B, L, H, -1).transpose(1, 2)
        keys = keys.view(B, S, H, -1).transpose(1, 2)
        values = values.view(B, S, H, -1).transpose(1, 2)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            queries,
            keys,
            values,
            dropout_p=self.dropout if self.training else 0,
            attn_mask=attn_mask,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, L, -1)
        return self.out_projection(attn_output)


class TransEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_head,
        ffn_hidden_size,
        dropout=0.1,
        activation="GELU",
    ):
        super(TransEncoderLayer, self).__init__()
        self.self_attn = AttentionLayer(hidden_size, n_head, dropout)

        self.hidden_size = hidden_size

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.ffn = MlpFFN(hidden_size, ffn_hidden_size, activation, dropout)

        assert activation in ACTIVATIONS, f"{activation} is not in {ACTIVATIONS}"
        self.activation = getattr(nn, activation)()

    def forward(self, target_x, exog_x, cross_mask=None, padding_mask=None):
        B, patch_nums, D = target_x.shape
        B_exog_num, _, _ = exog_x.shape
        exog_num = B_exog_num // B

        combined_x = torch.cat([target_x, exog_x], dim=0)
        residual_combined = combined_x

        combined_x = self.self_attn(
            combined_x, combined_x, combined_x, padding_mask=padding_mask, use_RoPE=True
        )
        combined_x = self.dropout(combined_x)
        combined_x = self.norm1(combined_x + residual_combined)
        target_x = combined_x[:B]
        exog_x = combined_x[B:]

        cls_target = target_x[:, -1, :].unsqueeze(1)
        cls_exog = exog_x.view(B, exog_num, patch_nums, D)[:, :, -1, :]
        cls_exog = torch.concat([cls_target, cls_exog], dim=1)
        residual_cls_target = cls_target
        cls_target = self.self_attn(
            cls_target, cls_exog, cls_exog, cross_mask=cross_mask
        )
        cls_target = self.dropout(cls_target)
        cls_target = self.norm2(cls_target + residual_cls_target)
        target_x = torch.concat([target_x[:, :-1, :], cls_target], dim=1)

        flatten_target_x = target_x.view(-1, D)
        flatten_exog_x = exog_x.view(-1, D)
        target_x_num = flatten_target_x.shape[0]
        x = torch.concat([flatten_target_x, flatten_exog_x], dim=0)
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + residual
        x = self.norm3(x)

        target_x = x[:target_x_num, :].view(B, patch_nums, D)
        exog_x = x[target_x_num:, :].view(B * exog_num, patch_nums, D)

        return target_x, exog_x


class TransEncoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_head,
        patch_size,
        ffn_hidden_size,
        encoder_layer_num,
        dropout=0.1,
        activation="GELU",
    ):
        super(TransEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList(
            [
                TransEncoderLayer(
                    hidden_size,
                    n_head,
                    ffn_hidden_size,
                    dropout,
                    activation,
                )
                for l in range(encoder_layer_num)
            ]
        )
        self.hidden_size = hidden_size
        self.target_token = nn.Parameter(torch.zeros(1, hidden_size))
        self.exog_token = nn.Parameter(torch.zeros(1, hidden_size))
        nn.init.xavier_normal_(self.target_token)
        nn.init.xavier_normal_(self.exog_token)
        self.patch_embedding = TokenEmbedding(hidden_size, patch_size, patch_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, channel_nums, cross_mask=None, padding_mask=None):
        B_C, _, _ = x.shape
        B = B_C // channel_nums
        x = self.patch_embedding(x)
        x = self.dropout(x)
        patch_nums = x.shape[1]
        x = x.view(B, channel_nums, patch_nums, self.hidden_size)
        exog_nums = channel_nums - 1

        target_x = x[:, 0, :, :]
        exog_x = x[:, 1:, :, :]

        target_token = self.target_token.unsqueeze(0).expand(B, -1, -1)
        exog_token = (
            self.exog_token.unsqueeze(0).unsqueeze(0).expand(B, exog_nums, -1, -1)
        )

        target_x = torch.cat([target_x, target_token], dim=1)
        exog_x = torch.cat([exog_x, exog_token], dim=2)
        exog_x = exog_x.reshape(B * exog_nums, patch_nums + 1, self.hidden_size)

        for encoder_layer in self.encoder_layers:
            target_x, exog_x = encoder_layer(
                target_x, exog_x, cross_mask=cross_mask, padding_mask=padding_mask
            )

        target_x = self.norm(target_x)
        return target_x[:, -1, :]


def masked_mean(x, mask, dim=-1, keepdim=True):
    x_nan = x.masked_fill(mask < 1, float("nan"))
    x_mean = x_nan.nanmean(dim=dim, keepdim=keepdim)
    x_mean = torch.nan_to_num(x_mean, nan=0.0)
    return x_mean


def std_statistics(x, mask, dim=-1, eps=1e-6):
    x_means = masked_mean(x=x, mask=mask, dim=dim)
    x_stds = torch.sqrt(masked_mean(x=(x - x_means) ** 2, mask=mask, dim=dim))

    x_stds[x_stds == 0] = 1.0
    x_stds = x_stds + eps
    return x_means, x_stds


def std_scaler(x, x_means, x_stds):
    return (x - x_means) / x_stds


def inv_std_scaler(z, x_mean, x_std):
    return (z * x_std) + x_mean


class TemporalNorm(nn.Module):
    def __init__(self, scaler_type="standard", dim=-1, eps=1e-6, num_features=None):
        super().__init__()
        compute_statistics = {
            "standard": std_statistics,
        }
        scalers = {
            "standard": std_scaler,
        }
        inverse_scalers = {
            "standard": inv_std_scaler,
        }
        assert scaler_type in scalers.keys(), f"{scaler_type} not defined"
        if (scaler_type == "revin") and (num_features is None):
            raise Exception("You must pass num_features for ReVIN scaler.")

        self.compute_statistics = compute_statistics[scaler_type]
        self.scaler = scalers[scaler_type]
        self.inverse_scaler = inverse_scalers[scaler_type]
        self.scaler_type = scaler_type
        self.dim = dim
        self.eps = eps

        if scaler_type == "revin":
            self._init_params(num_features=num_features)

    def _init_params(self, num_features):
        if self.dim == 1:
            self.revin_bias = nn.Parameter(torch.zeros(1, 1, num_features))
            self.revin_weight = nn.Parameter(torch.ones(1, 1, num_features))
        elif self.dim == -1:
            self.revin_bias = nn.Parameter(torch.zeros(1, num_features, 1))
            self.revin_weight = nn.Parameter(torch.ones(1, num_features, 1))

    def transform(self, x, mask):
        """Center and scale the data.

        Args
            x: torch.Tensor shape [batch, time, channels].<br>
            mask: torch Tensor bool, shape  [batch, time] where x is valid and False
                    where `x` should be masked. Mask should not be all False in any column of
                    dimension dim to avoid NaNs from zero division.<br>

        Returns:
            z: torch.Tensor same shape as x, except scaled.
        """
        x_shift, x_scale = self.compute_statistics(
            x=x, mask=mask, dim=self.dim, eps=self.eps
        )
        self.x_shift = x_shift
        self.x_scale = x_scale

        if self.scaler_type == "revin":
            self.x_shift = self.x_shift + self.revin_bias
            self.x_scale = self.x_scale * (torch.relu(self.revin_weight) + self.eps)

        z = self.scaler(x, x_shift, x_scale)
        return z

    def inverse_transform(self, z, x_shift=None, x_scale=None):
        """Scale back the data to the original representation.

        Args:
            z: torch.Tensor shape [batch, time, channels], scaled.<br>

        Returns:
            x: torch.Tensor original data.
        """

        if x_shift is None:
            x_shift = self.x_shift
        if x_scale is None:
            x_scale = self.x_scale

        x = self.inverse_scaler(z, x_shift, x_scale)
        return x

    def forward(self, x):
        pass
