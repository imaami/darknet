
#ifndef RNN_LAYER_H
#define RNN_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

layer_t make_rnn_layer(int batch, int inputs, int hidden, int outputs, int steps, ACTIVATION activation, int batch_normalize, int log);

void forward_rnn_layer(layer_t l, network_state state);
void backward_rnn_layer(layer_t l, network_state state);
void update_rnn_layer(layer_t l, int batch, float learning_rate, float momentum, float decay);

#ifdef GPU
void forward_rnn_layer_gpu(layer_t l, network_state state);
void backward_rnn_layer_gpu(layer_t l, network_state state);
void update_rnn_layer_gpu(layer_t l, int batch, float learning_rate, float momentum, float decay);
void push_rnn_layer(layer_t l);
void pull_rnn_layer(layer_t l);
#endif

#endif

