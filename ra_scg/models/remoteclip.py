from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision.transforms import Compose


class RemoteCLIPAttackBridge(nn.Module):
    """
    Exposes RemoteCLIP through the interfaces expected by
    Co-Attack, SGA, DRA and SA-AET.

    Attack images are stored in raw RGB [0,1] pixel space.
    inference_image() expects model-normalized tensors, matching
    the original attack repositories.
    """

    def __init__(
        self,
        checkpoint: Path,
        device: torch.device,
        architecture: str = "ViT-B-32",
        bert_tokenizer: Any | None = None,
        capitalize_text: bool = True,
    ) -> None:
        super().__init__()

        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

        model, _, preprocess = open_clip.create_model_and_transforms(
            architecture,
            pretrained="openai",
            device="cpu",
            cache_dir="cache/weights/open_clip",
        )

        state = torch.load(checkpoint, map_location="cpu")

        if isinstance(state, dict):
            for candidate in ("state_dict", "model"):
                if candidate in state and isinstance(state[candidate], dict):
                    state = state[candidate]
                    break

        if not isinstance(state, dict):
            raise TypeError(
                f"Unsupported checkpoint type: {type(state)}"
            )

        state = {
            key.removeprefix("module."): value
            for key, value in state.items()
        }

        message = model.load_state_dict(state, strict=True)
        print("Checkpoint load:", message)

        # Target of optimization is the input, not model parameters.
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        self.model = model.to(device).float().eval()
        self.architecture = architecture
        self.bert_tokenizer = bert_tokenizer
        self.capitalize_text = capitalize_text

        self.clip_tokenizer = open_clip.get_tokenizer(
            architecture
        )

        # Validated OpenCLIP preprocessing:
        # resize/crop/to-tensor, followed by Normalize.
        self.raw_preprocess = Compose(
            preprocess.transforms[:-1]
        )
        self.normalization = preprocess.transforms[-1]

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def visual(self):
        return self.model.visual

    @property
    def logit_scale(self):
        return self.model.logit_scale

    def load_raw_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.raw_preprocess(image)

        if tensor.ndim != 3:
            raise RuntimeError(
                f"Unexpected image shape: {tuple(tensor.shape)}"
            )

        if tensor.min() < 0 or tensor.max() > 1:
            raise RuntimeError(
                "Raw image preprocessing did not produce [0,1]"
            )

        return tensor

    def _prepare_texts(
        self,
        text_input: Sequence[str] | Any,
    ) -> list[str]:
        if isinstance(text_input, (list, tuple)):
            texts = [str(text) for text in text_input]

        elif hasattr(text_input, "input_ids"):
            if self.bert_tokenizer is None:
                raise RuntimeError(
                    "BERT BatchEncoding was supplied, but "
                    "bert_tokenizer is not configured."
                )

            texts = self.bert_tokenizer.batch_decode(
                text_input.input_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        else:
            raise TypeError(
                f"Unsupported text input type: {type(text_input)}"
            )

        normalized: list[str] = []

        for text in texts:
            text = " ".join(text.strip().split())

            if self.capitalize_text:
                text = text.capitalize()

            normalized.append(text)

        return normalized

    def encode_image_normalized(
        self,
        image_normalized: torch.Tensor,
    ) -> torch.Tensor:
        features = self.model.encode_image(
            image_normalized
        )
        return F.normalize(features.float(), dim=-1)

    def encode_text_strings(
        self,
        texts: Sequence[str],
    ) -> torch.Tensor:
        texts = self._prepare_texts(texts)

        tokens = self.clip_tokenizer(texts).to(
            self.device
        )

        features = self.model.encode_text(tokens)
        return F.normalize(features.float(), dim=-1)

    def inference_image(
        self,
        image_normalized: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = self.encode_image_normalized(
            image_normalized
        )

        return {
            "image_embed": features,
            "image_feat": features,
        }

    def inference_text(
        self,
        text_input: Sequence[str] | Any,
    ) -> dict[str, torch.Tensor]:
        texts = self._prepare_texts(text_input)

        tokens = self.clip_tokenizer(texts).to(
            self.device
        )

        features = F.normalize(
            self.model.encode_text(tokens).float(),
            dim=-1,
        )

        return {
            "text_embed": features,
            "text_feat": features,
        }

    def inference(
        self,
        image_normalized: torch.Tensor,
        text_input: Sequence[str] | Any,
    ) -> dict[str, torch.Tensor]:
        image_features = self.inference_image(
            image_normalized
        )["image_feat"]

        text_features = self.inference_text(
            text_input
        )["text_feat"]

        return {
            "image_feat": image_features,
            "text_feat": text_features,
        }
