from __future__ import annotations

import torch

from .core import (
    RASCGAblationAttacker,
    SELECTED_LAYERS,
    VIEW_NAMES,
)

from .phrase_relevance import (
    RemoteCLIPPatchPhraseRelevance,
)

from .semantic_guidance import (
    build_semantic_guidance_batch,
)

from .utils.nltk_resources import (
    require_pos_tagger,
)


class RASCGAttack:
    """
    Paper-faithful RA-SCG attack.

    The public class exposes the final Full configuration only:

    - five views;
    - relation residual enabled;
    - confidence-weighted relation residual;
    - directional consensus enabled;
    - semantic quality enabled;
    - relation lambda = 0.5;
    - epsilon = 2/255;
    - step size = 0.5/255;
    - ten image updates.

    Frozen adversarial captions must be supplied before image
    optimization. No live text candidate search occurs inside
    run_batch().
    """

    def __init__(
        self,
        bridge,
        *,
        nltk_data=None,
    ):
        require_pos_tagger(
            nltk_data
        )

        self.bridge = bridge

        self.extractor = (
            RemoteCLIPPatchPhraseRelevance(
                bridge=bridge,
                selected_layers=SELECTED_LAYERS,
            )
        )

        self.core = (
            RASCGAblationAttacker(
                use_relation=True,
                use_relation_confidence=True,
                use_directional_consensus=True,
                use_semantic_quality=True,
                active_views=VIEW_NAMES,
                relation_lambda=0.5,
                epsilon=2 / 255,
                step_size=0.5 / 255,
                steps=10,
                alpha=3.0,
            )
        )

    def run_batch(
        self,
        clean_images: torch.Tensor,
        clean_caption_groups: list[list[str]],
        frozen_adversarial_caption_groups: list[list[str]],
    ):
        """
        Args
        ----
        clean_images:
            Tensor [B,3,H,W] in [0,1].

        clean_caption_groups:
            B groups of exactly five clean paired captions.

        frozen_adversarial_caption_groups:
            B groups of exactly five adversarial captions selected
            offline before image optimization.

        Returns
        -------
        dict
            adv_images
            history
            semantic_centers
            R
            C
            F
            entity_maps
            spatial_maps
            relation_valid
        """

        batch = len(
            clean_images
        )

        if len(
            clean_caption_groups
        ) != batch:
            raise ValueError(
                "clean_caption_groups batch mismatch"
            )

        if len(
            frozen_adversarial_caption_groups
        ) != batch:
            raise ValueError(
                "frozen_adversarial_caption_groups "
                "batch mismatch"
            )

        for group in clean_caption_groups:
            if len(group) != 5:
                raise ValueError(
                    "Each image must have exactly "
                    "five clean captions."
                )

        for group in (
            frozen_adversarial_caption_groups
        ):
            if len(group) != 5:
                raise ValueError(
                    "Each image must have exactly "
                    "five frozen adversarial captions."
                )

        (
            R,
            C,
            Fmap,
            entity_maps,
            spatial_maps,
            relation_valid,
        ) = build_semantic_guidance_batch(
            self.extractor,
            clean_images,
            clean_caption_groups,
        )

        flat_adv_text = [
            caption
            for group
            in frozen_adversarial_caption_groups
            for caption in group
        ]

        with torch.no_grad():
            text_features = (
                self.bridge
                .encode_text_strings(
                    flat_adv_text
                )
                .float()
                .reshape(
                    batch,
                    5,
                    -1,
                )
                .detach()
                .clone()
            )

        (
            adv,
            history,
            centers,
        ) = self.core.run(
            self.bridge,
            clean_images,
            text_features,
            R,
            C,
            Fmap,
            entity_maps,
            spatial_maps,
            relation_valid,
        )

        return {
            "adv_images": adv,
            "history": history,
            "semantic_centers": centers,
            "R": R,
            "C": C,
            "F": Fmap,
            "entity_maps": entity_maps,
            "spatial_maps": spatial_maps,
            "relation_valid": relation_valid,
        }
