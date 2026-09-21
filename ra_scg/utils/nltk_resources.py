from __future__ import annotations

import os
from pathlib import Path


RESOURCE = (
    "taggers/"
    "averaged_perceptron_tagger_eng"
)


def configure_nltk_data(
    path: str | Path | None = None,
) -> Path | None:
    import nltk

    if path is None:
        value = os.environ.get(
            "NLTK_DATA"
        )

        if not value:
            return None

        path = value.split(
            os.pathsep
        )[0]

    path = (
        Path(path)
        .expanduser()
        .resolve()
    )

    if str(path) not in nltk.data.path:
        nltk.data.path.insert(
            0,
            str(path),
        )

    return path


def require_pos_tagger(
    nltk_data: str | Path | None = None,
) -> None:
    import nltk

    configure_nltk_data(
        nltk_data
    )

    try:
        nltk.data.find(
            RESOURCE
        )

        nltk.pos_tag(
            [
                "remote",
                "sensing",
                "image",
            ]
        )

    except LookupError as exc:
        raise RuntimeError(
            "RA-SCG requires the NLTK English POS tagger "
            "'averaged_perceptron_tagger_eng'.\n\n"
            "Install it with:\n"
            "  python scripts/setup_nltk.py --download\n\n"
            "Then either export NLTK_DATA or pass nltk_data "
            "to RASCGAttack."
        ) from exc
