def test_layer_registry():
    from models.layers import layer_type_registry

    assert len(layer_type_registry) > 0
