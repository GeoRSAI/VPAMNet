import os
import pydoc
from typing import List, Tuple, Type, Union

import torch
import torch.nn.functional as F
from dynamic_network_architectures.building_blocks.helper import (
    convert_conv_op_to_dim,
    get_matching_convtransp,
)
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dinounet.dinov3.eval.segmentation.models.backbone.dinov3_adapter import DINOv3_Adapter
from dinounet.dinov3.hub.backbones import (
    dinov3_vit7b16,
    dinov3_vitb16,
    dinov3_vitl16,
    dinov3_vits16,
)
from dinounet.dinov3.models.vision_transformer import DinoVisionTransformer


DINOv3_MODEL_FACTORIES = {
    "dinounet_s": dinov3_vits16,
    "dinounet_b": dinov3_vitb16,
    "dinounet_l": dinov3_vitl16,
    "dinounet_7b": dinov3_vit7b16,
}

DINOv3_INTERACTION_INDEXES = {
    "dinounet_s": [2, 5, 8, 11],
    "dinounet_b": [2, 5, 8, 11],
    "dinounet_l": [4, 11, 17, 23],
    "dinounet_7b": [9, 19, 29, 39],
}

DINOv3_MODEL_INFO = {
    "dinounet_s": {"embed_dim": 384, "depth": 12, "num_heads": 6, "params": "~22M"},
    "dinounet_b": {"embed_dim": 768, "depth": 12, "num_heads": 12, "params": "~86M"},
    "dinounet_l": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "params": "~300M"},
    "dinounet_7b": {"embed_dim": 4096, "depth": 40, "num_heads": 32, "params": "~7B"},
}


def load_dinov3_model(model_name: str, pretrained_path: str = None) -> DinoVisionTransformer:
    if model_name not in DINOv3_MODEL_FACTORIES:
        supported_models = list(DINOv3_MODEL_FACTORIES.keys())
        raise ValueError(f"Unsupported model: {model_name}. Supported models: {supported_models}")

    model_factory = DINOv3_MODEL_FACTORIES[model_name]
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading custom pretrained weights from {pretrained_path}")
        model = model_factory(pretrained=False)
        state_dict = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        print("Successfully loaded custom pretrained weights")
    else:
        print(f"Loading default pretrained weights for {model_name}")
        model = model_factory(pretrained=True)
        print("Successfully loaded default pretrained weights")

    return model


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.fc(w)
        return x * w


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        norm: Type[nn.Module] = nn.BatchNorm2d,
        act: Type[nn.Module] = nn.ReLU,
        norm_kwargs: dict = None,
        act_kwargs: dict = None,
    ):
        super().__init__()
        norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        act_kwargs = {"inplace": True} if act_kwargs is None else act_kwargs
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_ch, bias=bias
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
        self.bn = norm(out_ch, **norm_kwargs) if norm is not None else nn.Identity()
        self.act = act(**act_kwargs) if act is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class LearnableUpsampleBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.up2 = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2, bias=True)

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        out = x
        while h * 2 <= target_size[0] and w * 2 <= target_size[1]:
            out = self.up2(out)
            h, w = out.shape[2], out.shape[3]
        if (h, w) != target_size:
            out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
        return out


class FAPM(nn.Module):
    def __init__(
        self,
        in_ch: int,
        rank: int,
        out_ch_list: List[int],
        norm: Type[nn.Module] = nn.BatchNorm2d,
        act: Type[nn.Module] = nn.ReLU,
        norm_kwargs: dict = None,
        act_kwargs: dict = None,
        bias: bool = False,
    ):
        super().__init__()
        norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        act_kwargs = {"inplace": True} if act_kwargs is None else act_kwargs

        self.shared_basis = nn.Conv2d(in_ch, rank, kernel_size=1, bias=bias)
        self.specific_bases = nn.ModuleList([nn.Conv2d(in_ch, rank, kernel_size=1, bias=bias) for _ in out_ch_list])
        self.film_generators = nn.ModuleList(
            [nn.Conv2d(rank, rank * 2, kernel_size=1, bias=bias) for _ in out_ch_list]
        )
        self.refinement_blocks = nn.ModuleList()
        self.shortcut_projections = nn.ModuleList()

        for oc in out_ch_list:
            reduce = nn.Conv2d(rank, oc, kernel_size=1, bias=bias)
            dw = DepthwiseSeparableConv(
                oc,
                oc,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
                norm=norm,
                act=act,
                norm_kwargs=norm_kwargs,
                act_kwargs=act_kwargs,
            )
            refine = nn.Conv2d(oc, oc, kernel_size=1, bias=bias)
            se = SqueezeExcitation(oc)

            self.refinement_blocks.append(
                nn.Sequential(
                    reduce,
                    norm(oc, **norm_kwargs) if norm is not None else nn.Identity(),
                    act(**act_kwargs) if act is not None else nn.Identity(),
                    dw,
                    refine,
                    se,
                )
            )
            if rank != oc:
                self.shortcut_projections.append(nn.Conv2d(rank, oc, kernel_size=1, bias=bias))
            else:
                self.shortcut_projections.append(nn.Identity())

    def forward(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        out = []
        for i, x in enumerate(x_list):
            z_shared = self.shared_basis(x)
            z_specific = self.specific_bases[i](x)
            gamma_beta = self.film_generators[i](z_shared)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
            z_modulated = gamma * z_specific + beta
            refined = self.refinement_blocks[i](z_modulated)
            shortcut = self.shortcut_projections[i](z_modulated)
            out.append(refined + shortcut)
        return out


class DINOv3EncoderAdapter(nn.Module):
    def __init__(
        self,
        dinov3_adapter: DINOv3_Adapter,
        target_channels: List[int],
        adapter_type: str = "default",
        rank: int = 256,
        conv_op: Type[_ConvNd] = nn.Conv2d,
        norm_op: Union[None, Type[nn.Module]] = nn.BatchNorm2d,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = nn.ReLU,
        nonlin_kwargs: dict = None,
        conv_bias: bool = False,
    ):
        super().__init__()
        self.dinov3_adapter = dinov3_adapter
        self.target_channels = target_channels
        self.conv_op = conv_op
        self.norm_op = norm_op if norm_op is not None else nn.BatchNorm2d
        self.norm_op_kwargs = norm_op_kwargs if norm_op_kwargs is not None else {}
        self.nonlin = nonlin if nonlin is not None else nn.ReLU
        self.nonlin_kwargs = nonlin_kwargs if nonlin_kwargs is not None else {"inplace": True}
        self.conv_bias = conv_bias
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs

        in_ch = self.dinov3_adapter.backbone.embed_dim
        self.fapm = FAPM(
            in_ch,
            rank,
            target_channels,
            norm=self.norm_op,
            act=self.nonlin,
            norm_kwargs=self.norm_op_kwargs,
            act_kwargs=self.nonlin_kwargs,
            bias=conv_bias,
        )

        self.ups = nn.ModuleList([LearnableUpsampleBlock(oc) for oc in target_channels])
        self.output_channels = target_channels
        self.strides = [[2, 2]] * len(target_channels)
        self.kernel_sizes = [[3, 3]] * len(target_channels)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        _, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)
        elif c != 3:
            if c < 3:
                x = x.repeat(1, 3 // c + (1 if 3 % c != 0 else 0), 1, 1)[:, :3, :, :]
            else:
                x = x[:, :3, :, :]

        feats = self.dinov3_adapter(x)
        keys = ["1", "2", "3", "4"]
        x_list = [feats[k] for k in keys]
        ys = self.fapm(x_list)

        skips = []
        for i, y in enumerate(ys):
            target = (h // (2 ** i), w // (2 ** i))
            skips.append(self.ups[i](y, target))
        return skips

    def compute_conv_feature_map_size(self, input_size):
        return 0


class UNetDecoder(nn.Module):
    def __init__(
        self,
        encoder: PlainConvEncoder,
        num_classes: int,
        n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
        deep_supervision,
        nonlin_first: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        conv_bias: bool = None,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, (
            "n_conv_per_stage must have as many entries as we have resolution stages - 1 "
            f"(n_stages in encoder - 1), here: {n_stages_encoder}"
        )

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs

        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(
                transpconv_op(
                    input_features_below,
                    input_features_skip,
                    stride_for_transpconv,
                    stride_for_transpconv,
                    bias=conv_bias,
                )
            )
            stages.append(
                StackedConvBlocks(
                    n_conv_per_stage[s - 1],
                    encoder.conv_op,
                    2 * input_features_skip,
                    input_features_skip,
                    encoder.kernel_sizes[-(s + 1)],
                    1,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first,
                )
            )
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]
        return seg_outputs[0] if not self.deep_supervision else seg_outputs


class DinoUNet(nn.Module):
    def __init__(
        self,
        network_config: dict = None,
        input_channels: int = None,
        num_classes: int = None,
        dinov3_pretrained_path: str = "dinounet/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        dinov3_model_name: str = "dinounet_s",
        adapter_type: str = "default",
        n_stages: int = None,
        features_per_stage: Union[int, List[int], Tuple[int, ...]] = None,
        conv_op: Type[_ConvNd] = None,
        kernel_sizes: Union[int, List[int], Tuple[int, ...]] = None,
        strides: Union[int, List[int], Tuple[int, ...]] = None,
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]] = None,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]] = None,
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
    ):
        super().__init__()

        if network_config is not None:
            arch = network_config["architecture"]

            def _resolve_op(op_str):
                if op_str is None:
                    return None
                if isinstance(op_str, str):
                    return pydoc.locate(op_str)
                return op_str

            input_channels = input_channels or 3
            self.adapter_type = adapter_type
            num_classes = num_classes or 2
            n_stages = arch["n_stages"]
            features_per_stage = arch["features_per_stage"]
            conv_op = _resolve_op(arch["conv_op"])
            kernel_sizes = arch["kernel_sizes"]
            strides = arch["strides"]
            n_conv_per_stage = arch["n_conv_per_stage"]
            n_conv_per_stage_decoder = arch["n_conv_per_stage_decoder"]
            conv_bias = arch.get("conv_bias", False)
            norm_op = _resolve_op(arch["norm_op"])
            norm_op_kwargs = arch.get("norm_op_kwargs", {})
            dropout_op = _resolve_op(arch["dropout_op"])
            dropout_op_kwargs = arch.get("dropout_op_kwargs", {})
            nonlin = _resolve_op(arch["nonlin"])
            nonlin_kwargs = arch.get("nonlin_kwargs", {})
            deep_supervision = arch.get("deep_supervision", False)
            nonlin_first = arch.get("nonlin_first", False)

        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        if n_stages != 4:
            print(f"Warning: DINOv3_Adapter outputs 4 scales, but n_stages={n_stages}. Adjusting to 4.")
            n_stages = 4
            if isinstance(features_per_stage, int):
                features_per_stage = [features_per_stage * (2 ** i) for i in range(4)]
            elif len(features_per_stage) != 4:
                base_features = features_per_stage[0] if features_per_stage else 32
                features_per_stage = [base_features * (2 ** i) for i in range(4)]

        self.encoder = self._create_dinov3_encoder(
            dinov3_pretrained_path,
            dinov3_model_name,
            features_per_stage,
            conv_op,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            conv_bias,
            adapter_type,
        )
        self.decoder = UNetDecoder(
            self.encoder,
            num_classes,
            n_conv_per_stage_decoder,
            deep_supervision,
            nonlin_first=nonlin_first,
        )

    def _create_dinov3_encoder(
        self,
        pretrained_path,
        model_name,
        features_per_stage,
        conv_op,
        norm_op,
        norm_op_kwargs,
        dropout_op,
        dropout_op_kwargs,
        nonlin,
        nonlin_kwargs,
        conv_bias,
        adapter_type="default",
    ):
        if model_name not in DINOv3_MODEL_INFO:
            raise ValueError(f"Unknown model: {model_name}")

        model_info = DINOv3_MODEL_INFO[model_name]
        interaction_indexes = DINOv3_INTERACTION_INDEXES[model_name]

        print(f"🔧 Creating DINOv3 encoder: {model_name}")
        print(f"   Embedding dimension: {model_info['embed_dim']}")
        print(f"   Model depth: {model_info['depth']}")
        print(f"   Number of attention heads: {model_info['num_heads']}")
        print(f"   Parameter count: {model_info['params']}")
        print(f"   Interaction layer indices: {interaction_indexes}")

        dinov3_backbone = load_dinov3_model(model_name, pretrained_path)
        dinov3_adapter = DINOv3_Adapter(
            backbone=dinov3_backbone,
            interaction_indexes=interaction_indexes,
            pretrain_size=512,
            conv_inplane=64,
            n_points=4,
            deform_num_heads=16,
            drop_path_rate=0.3,
            init_values=0.0,
            with_cffn=True,
            cffn_ratio=0.25,
            deform_ratio=0.5,
            add_vit_feature=True,
            use_extra_extractor=True,
            with_cp=True,
        )

        return DINOv3EncoderAdapter(
            dinov3_adapter=dinov3_adapter,
            target_channels=features_per_stage,
            conv_op=conv_op,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            conv_bias=conv_bias,
        )

    def forward(self, x):
        skips = self.encoder(x)
        output = self.decoder(skips)
        return output

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "just give the image size without color/feature channels or batch channel. "
            "Do not give input_size=(b, c, x, y(, z)). Give input_size=(x, y(, z))!"
        )
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size
        )

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)

    @classmethod
    def from_config(
        cls,
        network_config: dict,
        input_channels: int,
        num_classes: int,
        dinov3_pretrained_path: str = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        dinov3_model_name: str = "dinounet_s",
    ):
        return cls(
            network_config=network_config,
            input_channels=input_channels,
            num_classes=num_classes,
            dinov3_pretrained_path=dinov3_pretrained_path,
            dinov3_model_name=dinov3_model_name,
        )

if __name__ == "__main__":

    from fvcore.nn import FlopCountAnalysis,parameter_count
    # 1. 实例化模型并设为评估模式
    config = DinoUNetTrainer._network_config.copy()
    config['architecture'] = config['architecture'].copy()
    config['architecture']['deep_supervision'] = enable_deep_supervision
    model = DinoUNet(input_channels=3,
                     num_classes=16,
                     dinov3_pretrained_path='/media/csu/sda2/xjwDLexp/exp011_wavelet/pretrain/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
                     dinov3_model_name="dinounet_s",
                     features_per_stage=4,
                     n_conv_per_stage=2)
    model.eval()
    input_tensor = torch.randn(1, 3, 512, 512)

    # 计算
    flops = FlopCountAnalysis(model, input_tensor).total()
    params = sum(parameter_count(model).values())
    # 输出 (单位: G 和 M)
    print(f"FLOPs: {flops / 1e9:.2f} G")
    print(f"Params: {params / 1e6:.2f} M")