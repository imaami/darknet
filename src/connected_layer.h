#ifndef CONNECTED_LAYER_H
#define CONNECTED_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize);

void forward_connected_layer(layer layer, network_state state);
void backward_connected_layer(layer layer, network_state state);
void update_connected_layer(layer layer, int batch, float learning_rate, float momentum, float decay);

#ifdef GPU
void forward_connected_layer_gpu(layer layer, network_state state);
void backward_connected_layer_gpu(layer layer, network_state state);
void update_connected_layer_gpu(layer layer, int batch, float learning_rate, float momentum, float decay);
void push_connected_layer(layer layer);
void pull_connected_layer(layer layer);
#endif

#endif

