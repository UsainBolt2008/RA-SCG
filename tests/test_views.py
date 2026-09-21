import torch
from ra_scg.views import VIEW_NAMES, build_views

def test_five_views_and_shapes():
    x=torch.rand(2,3,64,64)
    centers=torch.zeros(2,2)
    views=build_views(x,centers)
    assert tuple(views)==tuple(VIEW_NAMES)
    assert len(views)==5
    assert torch.equal(views['identity'],x)
    for y in views.values():
        assert y.shape==x.shape
