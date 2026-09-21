# Paper-to-code map

The public implementation is refactored from the experimentally frozen
RA-SCG Full path without changing its mathematical operations.

| Paper component | Public implementation |
|---|---|
| Five semantic views | `ra_scg/views.py` |
| ViT layers 2,5,8,11 | `ra_scg/core.py::SELECTED_LAYERS` |
| R, C, F semantic guidance | `ra_scg/semantic_guidance.py` |
| Entity/spatial relation support | `ra_scg/core.py` |
| Base gradient | `ra_scg/core.py` |
| Relation gradient | `ra_scg/core.py` |
| Relation residual | `ra_scg/core.py` |
| Directional consensus d_i | `ra_scg/core.py` |
| Semantic quality q_i | `ra_scg/core.py` |
| Weighted aggregate g_k | `ra_scg/core.py` |
| Momentum/sign/Linf update | `ra_scg/core.py` |
| Public attack API | `ra_scg/attack.py::RASCGAttack` |

The public default exposes only the final paper-facing RA-SCG
configuration.

Historical experiment names and abandoned gated variants are retained
only under `provenance/` and are not part of the public API.
