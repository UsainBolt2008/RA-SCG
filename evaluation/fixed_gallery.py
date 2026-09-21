from __future__ import annotations

import torch


def rank_from_scores(
    scores: torch.Tensor,
    positives,
) -> int:
    """
    Descending argsort with best-valid-positive rank.
    Rank is one-based.
    """

    order = torch.argsort(
        scores,
        descending=True,
    )

    inverse = torch.empty_like(
        order
    )

    inverse[order] = torch.arange(
        len(order),
        dtype=order.dtype,
        device=order.device,
    )

    positives = torch.as_tensor(
        positives,
        dtype=torch.long,
        device=order.device,
    )

    return (
        int(
            inverse[
                positives
            ].min().item()
        )
        + 1
    )


def ranks_from_score_matrix(
    scores: torch.Tensor,
    positive_indices,
) -> torch.Tensor:
    """
    Historical RSICD rank primitive.

    The complete query x gallery score matrix is constructed first,
    then a single 2-D argsort is executed along the gallery dimension.
    """
    if scores.ndim != 2:
        raise ValueError(
            f"expected 2-D score matrix, got shape={tuple(scores.shape)}"
        )
    if len(positive_indices) != int(scores.shape[0]):
        raise ValueError(
            "positive mapping cardinality must equal query count"
        )

    order = torch.argsort(
        scores,
        dim=1,
        descending=True,
    )
    inverse = torch.empty_like(order)
    positions = torch.arange(
        scores.shape[1],
        dtype=order.dtype,
        device=order.device,
    ).unsqueeze(0).expand_as(order)
    inverse.scatter_(1, order, positions)

    out = []
    for query_index, positives in enumerate(
        positive_indices
    ):
        p = torch.as_tensor(
            positives,
            dtype=torch.long,
            device=order.device,
        )
        out.append(
            int(
                inverse[
                    query_index,
                    p,
                ].min().item()
            )
            + 1
        )

    return torch.tensor(
        out,
        dtype=torch.long,
    )


def clean_i2t(
    image_features,
    text_features,
    positive_caption_indices,
    image_indices,
):
    return torch.tensor(
        [
            rank_from_scores(
                image_features[i]
                @ text_features.T,
                positive_caption_indices[i],
            )
            for i in image_indices
        ],
        dtype=torch.long,
    )


def clean_t2i(
    image_features,
    text_features,
    caption_to_image,
    caption_indices,
):
    return torch.tensor(
        [
            rank_from_scores(
                image_features
                @ text_features[c],
                [
                    int(
                        caption_to_image[c]
                    )
                ],
            )
            for c in caption_indices
        ],
        dtype=torch.long,
    )


def joint_i2t_fixed(
    adversarial_image_features,
    clean_text_features,
    adversarial_text_features,
    positive_caption_indices,
    image_indices,
):
    """
    Adversarial image query.

    Only its official positive captions are replaced with
    adversarial versions; non-paired gallery captions remain clean.
    """

    out = []

    for i in image_indices:
        positives = (
            positive_caption_indices[i]
        )

        scores = (
            adversarial_image_features[i]
            @ clean_text_features.T
        ).clone()

        p = torch.as_tensor(
            positives,
            dtype=torch.long,
            device=scores.device,
        )

        scores[p] = (
            adversarial_image_features[i]
            @ adversarial_text_features[p].T
        )

        out.append(
            rank_from_scores(
                scores,
                positives,
            )
        )

    return torch.tensor(
        out,
        dtype=torch.long,
    )


def joint_t2i_fixed(
    clean_image_features,
    adversarial_image_features,
    adversarial_text_features,
    caption_to_image,
    caption_indices,
):
    """
    Adversarial caption query.

    Only its paired positive image is adversarial; every non-paired
    gallery image remains clean.
    """

    out = []

    for c in caption_indices:
        positive = int(
            caption_to_image[c]
        )

        scores = (
            clean_image_features
            @ adversarial_text_features[c]
        ).clone()

        scores[positive] = (
            adversarial_image_features[
                positive
            ]
            @ adversarial_text_features[c]
        )

        out.append(
            rank_from_scores(
                scores,
                [positive],
            )
        )

    return torch.tensor(
        out,
        dtype=torch.long,
    )


def joint_i2t_fixed_matrix(
    adversarial_image_features,
    clean_text_features,
    adversarial_text_features,
    positive_caption_indices,
):
    """
    Historical RSICD joint I2T score construction.

    Construct the entire image-query x caption-gallery score matrix
    before ranking. Only the five paired positive captions of each
    query are replaced by adversarial-caption scores.
    """
    scores = (
        adversarial_image_features
        @ clean_text_features.T
    )

    for image_index, positives in enumerate(
        positive_caption_indices
    ):
        p = torch.as_tensor(
            positives,
            dtype=torch.long,
            device=scores.device,
        )
        scores[
            image_index,
            p,
        ] = (
            adversarial_image_features[
                image_index
            ]
            @ adversarial_text_features[p].T
        )

    return ranks_from_score_matrix(
        scores,
        positive_caption_indices,
    )


def joint_t2i_fixed_matrix(
    clean_image_features,
    adversarial_image_features,
    adversarial_text_features,
    caption_to_image,
):
    """
    Historical RSICD joint T2I score construction.

    Construct the entire caption-query x image-gallery score matrix
    before ranking. Only each query's paired positive image is
    replaced by its adversarial-image score.
    """
    cti = torch.as_tensor(
        caption_to_image,
        dtype=torch.long,
        device=clean_image_features.device,
    )

    scores = (
        adversarial_text_features
        @ clean_image_features.T
    )

    rows = torch.arange(
        len(cti),
        dtype=torch.long,
        device=scores.device,
    )

    scores[
        rows,
        cti,
    ] = (
        adversarial_text_features
        * adversarial_image_features[
            cti
        ]
    ).sum(-1)

    positives = [
        [int(v)]
        for v in cti.tolist()
    ]

    return ranks_from_score_matrix(
        scores,
        positives,
    )
