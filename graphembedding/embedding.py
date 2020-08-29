"""
Copyright 2020, All rights reserved.
Author : SangJae Kang
Mail : craftsangjae@gmail.com
"""
import pandas as pd
import numpy as np
from .dataset import TransEDataset, ComplExDataset
from .models import create_transEModel, create_complExModel


def transE(triplets:np.ndarray,
           embed_size=50,
           ord='l1',
           margin=1,
           learning_rate=2e-1,
           batch_size=1024,
           num_epochs=50,
           callbacks=None,
           keras_model=None,
           return_keras_model=False):
    # load dataset
    dataset = TransEDataset(triplets)

    # load Model
    if keras_model is None:
        model = create_transEModel(len(dataset.nodes), len(dataset.edges),
                                   embed_size, ord, margin, learning_rate)
    else:
        model = keras_model

    # Train transE Model
    try:
        for i in range(num_epochs):
            model.fit(dataset(batch_size),
                      epochs=i+1, initial_epoch=i,
                      callbacks=callbacks)
    except KeyboardInterrupt:
        pass

    # Extract Embedding
    node_embedding, edge_embedding = weight2embedding(model, dataset)

    if return_keras_model:
        return node_embedding, edge_embedding, model
    else:
        return node_embedding, edge_embedding


def complEx(triplets:np.ndarray,
            embed_size=50,
            n3_reg=1e-3,
            learning_rate=5e-1,
            num_negs=20,
            batch_size=1024,
            num_epochs=50,
            callbacks=None,
            keras_model=None,
            return_keras_model=False):
    # Load dataset
    dataset = ComplExDataset(triplets)

    # load Model
    if keras_model is None:
        model = create_complExModel(len(dataset.nodes), len(dataset.edges),
                                    embed_size, n3_reg, learning_rate)
    else:
        model = keras_model

    # Train complEx Model
    try:
        for i in range(num_epochs):
            model.fit(dataset(batch_size, num_negs),
                      epochs=i+1, initial_epoch=i,
                      class_weight={1: 1., 0: 1 / num_negs},
                      callbacks=callbacks)
    except KeyboardInterrupt:
        pass

    # Extract Embedding
    node_embedding, edge_embedding = weight2embedding(model, dataset)

    if return_keras_model:
        return node_embedding, edge_embedding, model
    else:
        return node_embedding, edge_embedding


def weight2embedding(model, dataset):
    node_embedding = pd.DataFrame(model.get_layer("node_embedding").get_weights()[0])
    node_embedding.index = dataset.nodes

    edge_embedding = pd.DataFrame(model.get_layer("edge_embedding").get_weights()[0])
    edge_embedding.index = dataset.edges
    return node_embedding, edge_embedding

