#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include "layer.h"
#include "network.h"

layer make_dropout_layer(int batch, int inputs, float probability);

void forward_dropout_layer(layer l, network_state state);
void backward_dropout_layer(layer l, network_state state);
void resize_dropout_layer(layer *l, int inputs);

#ifdef GPU
void forward_dropout_layer_gpu(layer l, network_state state);
void backward_dropout_layer_gpu(layer l, network_state state);

#endif
#endif
