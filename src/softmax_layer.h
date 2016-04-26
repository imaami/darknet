#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H
#include "layer.h"
#include "network.h"

void softmax_array(float *input, int n, float temp, float *output);
layer make_softmax_layer(int batch, int inputs, int groups);
void forward_softmax_layer(const layer l, network_state state);
void backward_softmax_layer(const layer l, network_state state);

#ifdef GPU
void pull_softmax_layer_output(const layer l);
void forward_softmax_layer_gpu(const layer l, network_state state);
void backward_softmax_layer_gpu(const layer l, network_state state);
#endif

#endif
