import torch

from alignit.models.alignnet import AlignNet


def test_alignnet_forward_shapes_cpu():
    model = AlignNet(backbone_name="resnet18", backbone_weights=None, use_vector_input=False, output_dim=7)
    model.eval()
    x = torch.randn(2, 3, 3, 64, 64)  # B=2, N=3 views
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 7)


def test_alignnet_with_vector_input():
    model = AlignNet(backbone_name="resnet18", backbone_weights=None, use_vector_input=True, output_dim=7)
    model.eval()
    x = torch.randn(1, 2, 3, 64, 64)
    vecs = [torch.randn(5)]
    with torch.no_grad():
        y = model(x, vecs)
    assert y.shape == (1, 7)


def test_alignnet_performance():
    model = AlignNet(backbone_name="efficientnet_b0", backbone_weights=None, use_vector_input=True, output_dim=7)
    model.eval()
    x = torch.randn(1, 3, 3, 224, 224)  # B=1, N=3 views
    vecs = [torch.randn(5)]
    with torch.no_grad():
        import time
        start_time = time.time()
        for _ in range(10):
            y = model(x, vecs)
        elapsed_time = (time.time() - start_time) / 10
        elapsed_time_ms = elapsed_time * 1000
        print(f"Forward pass took {elapsed_time_ms:.2f} ms")

    assert elapsed_time < 0.5
