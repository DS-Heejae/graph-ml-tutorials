"""
Copyright 2020, All rights reserved.
Author : SangJae Kang
Mail : craftsangjae@gmail.com
"""
import pandas as pd
import numpy as np
from .dataset import TransEDataset, ComplExDataset
from .layers import TransEScore, ComplexDotScore
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.initializers import RandomUniform, GlorotUniform
from tensorflow.keras.optimizers import Adagrad


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
    if keras_model is None:
        # build transE Model
        pos_sub_inputs = Input(shape=(), name='pos_subject')
        neg_sub_inputs = Input(shape=(), name='neg_subject')
        pos_obj_inputs = Input(shape=(), name='pos_object')
        neg_obj_inputs = Input(shape=(), name='neg_object')
        rel_inputs = Input(shape=(), name='relation')

        inputs = {
            "pos_subject": pos_sub_inputs,
            "neg_subject": neg_sub_inputs,
            "pos_object": pos_obj_inputs,
            "neg_object": neg_obj_inputs,
            "relation": rel_inputs
        }

        # 초기화 방식은 논문에 나와있는 방식으로 구성
        init_range = 6/np.sqrt(embed_size)
        init_op = RandomUniform(-init_range, init_range)

        node_layer = Embedding(input_dim=len(dataset.nodes),
                               output_dim=embed_size,
                               embeddings_initializer=init_op,
                               name='node_embedding')
        edge_layer = Embedding(input_dim=len(dataset.edges),
                               output_dim=embed_size,
                               embeddings_initializer=init_op,
                               name='edge_embedding')

        pos_sub = node_layer(pos_sub_inputs)
        neg_sub = node_layer(neg_sub_inputs)
        pos_obj = node_layer(pos_obj_inputs)
        neg_obj = node_layer(neg_obj_inputs)
        rel = edge_layer(rel_inputs)

        score = TransEScore(ord, margin)([pos_sub, neg_sub, pos_obj, neg_obj, rel])
        model = Model(inputs, score)

        # Compile transE Model
        model.add_loss(score)
        model.compile(optimizer=Adagrad(learning_rate))
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

    if keras_model is None:
        # Build complEx Model
        sub_inputs = Input(shape=(), name='subject')
        obj_inputs = Input(shape=(), name='object')
        rel_inputs = Input(shape=(), name='relation')
        inputs = {"subject": sub_inputs, "object": obj_inputs, "relation": rel_inputs}

        node_layer = Embedding(input_dim=len(dataset.nodes),
                               output_dim=embed_size,
                               embeddings_initializer=GlorotUniform(),
                               name='node_embedding')
        edge_layer = Embedding(input_dim=len(dataset.edges),
                               output_dim=embed_size,
                               embeddings_initializer=GlorotUniform(),
                               name='edge_embedding')

        sub_embed = node_layer(sub_inputs)
        rel_embed = edge_layer(rel_inputs)
        obj_embed = node_layer(obj_inputs)

        outputs = ComplexDotScore(n3_reg)([sub_embed, rel_embed, obj_embed])
        model = Model(inputs, outputs, name='complEx')

        # Compile complEx Model
        loss = BinaryCrossentropy(from_logits=True, reduction='sum')
        model.compile(optimizer=Adagrad(learning_rate), loss=loss, metrics=[loss])
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

