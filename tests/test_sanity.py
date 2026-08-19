def test_mlx_imports():
    import mlx.core as mx
    assert mx.array([1, 2]).sum().item() == 3

def test_workbench_imports():
    import workbench  # noqa: F401
