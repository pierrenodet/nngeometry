from nngeometry.layercollection import (
    Conv1dLayer,
    Conv2dLayer,
    EmbeddingLayer,
    LayerCollection,
    LinearLayer,
)


def last_layer(layer_collection: LayerCollection) -> LayerCollection:
    name = next(reversed(layer_collection.layers))
    last_layer_collection = LayerCollection()
    last_layer_collection.add_layer(name, layer_collection.layers[name])
    return last_layer_collection


def supported_kfac_layers(layer_collection: LayerCollection) -> LayerCollection:
    filtered_layer_collection = LayerCollection()
    for name, layer in layer_collection.layers.items():
        if isinstance(
            layer,
            (LinearLayer, Conv2dLayer, Conv1dLayer, EmbeddingLayer),
        ):
            filtered_layer_collection.add_layer(name, layer)
    return filtered_layer_collection
