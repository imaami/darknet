#ifndef LOCAL_LAYER_H
#define LOCAL_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

#ifdef GPU
void forward_local_layer_gpu(layer layer, network_state state);
void backward_local_layer_gpu(layer layer, network_state state);
void update_local_layer_gpu(layer layer, int batch, float learning_rate, float momentum, float decay);

void push_local_layer(layer layer);
void pull_local_layer(layer layer);
#endif

layer make_local_layer(int batch, int h, int w, int c, int n, int size, int stride, int pad, ACTIVATION activation);

void forward_local_layer(const layer layer, network_state state);
void backward_local_layer(layer layer, network_state state);
void update_local_layer(layer layer, int batch, float learning_rate, float momentum, float decay);

void bias_output(float *output, float *biases, int batch, int n, int size);
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size);

#endif

