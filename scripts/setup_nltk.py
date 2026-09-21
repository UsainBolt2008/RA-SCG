from __future__ import annotations

import argparse
import os
from pathlib import Path

import nltk


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--download",
        action="store_true",
        help=(
            "Download the required POS tagger "
            "if it is missing."
        ),
    )

    parser.add_argument(
        "--dir",
        type=Path,
        default=(
            Path(__file__)
            .resolve()
            .parents[1]
            / ".nltk_data"
        ),
        help=(
            "Directory in which NLTK resources "
            "are stored."
        ),
    )

    args = parser.parse_args()

    target = args.dir.resolve()

    target.mkdir(
        parents=True,
        exist_ok=True,
    )

    if str(target) not in nltk.data.path:
        nltk.data.path.insert(
            0,
            str(target),
        )

    os.environ[
        "NLTK_DATA"
    ] = str(target)

    resource = (
        "taggers/"
        "averaged_perceptron_tagger_eng"
    )

    try:
        found = nltk.data.find(
            resource,
            paths=[str(target)],
        )

        print(
            "NLTK_RESOURCE_STATUS="
            "PASS_EXISTING"
        )

        print(
            "RESOURCE=",
            found,
        )

    except LookupError:
        if not args.download:
            raise SystemExit(
                "Missing "
                "averaged_perceptron_tagger_eng.\n\n"
                "Run:\n"
                "  python scripts/setup_nltk.py --download"
            )

        print(
            "Downloading "
            "averaged_perceptron_tagger_eng ..."
        )

        ok = nltk.download(
            "averaged_perceptron_tagger_eng",
            download_dir=str(target),
            quiet=False,
            raise_on_error=False,
        )

        if not ok:
            raise RuntimeError(
                "NLTK download failed."
            )

        found = nltk.data.find(
            resource,
            paths=[str(target)],
        )

        print(
            "NLTK_RESOURCE_STATUS="
            "PASS_DOWNLOADED"
        )

        print(
            "RESOURCE=",
            found,
        )

    print(
        "POS_TAG=",
        nltk.pos_tag(
            [
                "white",
                "planes",
                "parked",
                "beside",
                "buildings",
            ]
        ),
    )

    print()
    print(
        "For a normal shell session:"
    )

    print(
        f'export NLTK_DATA="{target}"'
    )


if __name__ == "__main__":
    main()
