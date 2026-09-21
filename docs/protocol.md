# Frozen paper protocol

RA-SCG is evaluated as a transfer-based black-box attack.

During attack generation:

- only the active surrogate is used;
- victim parameters are unavailable;
- victim gradients are unavailable;
- no victim queries are used;
- no victim output feedback is used;
- the adversarial captions are fixed before image optimization;
- no live text-candidate generation or search is performed.

Image protocol:

- Linf epsilon: 2/255
- step size: 0.5/255
- image updates: 10
- outer batch size: 2
- seed: 0
- momentum coefficient: 1
- base-loss alpha: 3
- relation scale lambda: 0.5
- ViT blocks: 2, 5, 8, 11 (zero-based)

Five source views:

1. identity
2. global resize 0.75
3. global resize 1.25
4. semantic zoom 0.75
5. semantic zoom 0.50

Evaluation uses the paper's fixed-gallery conditional ASR protocol.
