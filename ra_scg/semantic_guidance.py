from __future__ import annotations

from collections import defaultdict

import torch

from .phrase_relevance import extract_phrase_views


def normalize_01(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return (
        (x - x.min())
        / (x.max() - x.min() + 1e-8)
    )


def consistency(
    maps: list[torch.Tensor],
) -> torch.Tensor:
    if len(maps) == 1:
        return torch.ones_like(
            maps[0]
        ).float()

    stack = torch.stack(
        [
            normalize_01(x)
            for x in maps
        ]
    )

    std = stack.std(
        dim=0,
        unbiased=False,
    )

    return 1.0 - normalize_01(std)


def build_semantic_guidance(
    extractor,
    image: torch.Tensor,
    captions: list[str],
):
    """
    Build paper-facing R, C, F and caption-specific
    entity/spatial maps from one clean image and its five
    clean paired captions.
    """

    if len(captions) != 5:
        raise ValueError(
            "RA-SCG expects exactly five clean paired captions; "
            f"received {len(captions)}"
        )

    phrases = []

    for caption_index, caption in enumerate(
        captions
    ):
        phrases.extend(
            extract_phrase_views(
                caption=caption,
                caption_index=caption_index,
            )
        )

    if not phrases:
        raise RuntimeError(
            "No phrase views were extracted "
            "from the clean captions."
        )

    layer_patch, _ = extractor.capture(
        image.unsqueeze(0)
    )

    phrase_feat = extractor.encode_phrases(
        phrases
    )

    layer_maps = {}
    type_pool = defaultdict(list)

    caption_pool = defaultdict(
        lambda: defaultdict(list)
    )

    for layer, patch in layer_patch.items():
        sim = patch[0] @ phrase_feat.T

        rows = []

        for j, phrase in enumerate(
            phrases
        ):
            m = normalize_01(
                sim[:, j]
            )

            rows.append(m)

            type_pool[
                phrase.kind
            ].append(m)

            caption_pool[
                int(
                    phrase.caption_index
                )
            ][
                phrase.kind
            ].append(m)

        layer_maps[
            int(layer)
        ] = (
            torch.stack(rows)
            .mean(0)
            .reshape(7, 7)
        )

    type_maps = {
        kind: (
            torch.stack(values)
            .mean(0)
            .reshape(7, 7)
        )
        for kind, values
        in type_pool.items()
    }

    R = normalize_01(
        (
            0.5
            * torch.stack(
                list(
                    type_maps.values()
                )
            ).mean(0)
            +
            0.5
            * torch.stack(
                list(
                    layer_maps.values()
                )
            ).mean(0)
        ).flatten()
    ).reshape(7, 7)

    layer_consistency = consistency(
        list(
            layer_maps.values()
        )
    )

    if len(type_maps) >= 2:
        type_consistency = consistency(
            list(
                type_maps.values()
            )
        )
    else:
        type_consistency = (
            torch.ones_like(R)
        )

    C = normalize_01(
        0.5 * layer_consistency
        +
        0.5 * type_consistency
    )

    Fmap = normalize_01(
        torch.sqrt(
            R.clamp(0, 1)
            * C.clamp(0, 1)
            + 1e-12
        )
    )

    entity_maps = []
    spatial_maps = []
    relation_valid = []

    for caption_index in range(5):
        entity = caption_pool[
            caption_index
        ].get(
            "entity",
            [],
        )

        spatial = caption_pool[
            caption_index
        ].get(
            "spatial",
            [],
        )

        if entity:
            entity_map = normalize_01(
                torch.stack(
                    entity
                ).mean(0)
            ).reshape(7, 7)
        else:
            entity_map = (
                torch.zeros_like(
                    Fmap
                )
            )

        if spatial:
            spatial_map = normalize_01(
                torch.stack(
                    spatial
                ).mean(0)
            ).reshape(7, 7)
        else:
            spatial_map = (
                torch.zeros_like(
                    Fmap
                )
            )

        entity_maps.append(
            entity_map
        )

        spatial_maps.append(
            spatial_map
        )

        relation_valid.append(
            bool(
                entity
                and spatial
            )
        )

    return (
        R.detach(),
        C.detach(),
        Fmap.detach(),
        torch.stack(
            entity_maps
        ).detach(),
        torch.stack(
            spatial_maps
        ).detach(),
        torch.tensor(
            relation_valid,
            device=image.device,
            dtype=torch.bool,
        ),
    )


def build_semantic_guidance_batch(
    extractor,
    clean_images: torch.Tensor,
    caption_groups: list[list[str]],
):
    """
    Convenience wrapper for a batch of clean images.

    Each clean image must have exactly five paired captions.
    """

    if len(clean_images) != len(
        caption_groups
    ):
        raise ValueError(
            "Number of clean images and caption groups must match."
        )

    rows = [
        build_semantic_guidance(
            extractor,
            image,
            captions,
        )
        for image, captions
        in zip(
            clean_images,
            caption_groups,
        )
    ]

    return tuple(
        torch.stack(
            [
                row[k]
                for row in rows
            ]
        )
        for k in range(6)
    )
