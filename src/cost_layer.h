#ifndef COST_LAYER_H
#define COST_LAYER_H
#include "layer.h"
#include "network.h"

COST_TYPE get_cost_type(char *s);
char *get_cost_string(COST_TYPE a);
layer make_cost_layer(int batch, int inputs, COST_TYPE type, float scale);
void forward_cost_layer(const layer l, network_state state);
void backward_cost_layer(const layer l, network_state state);
void resize_cost_layer(layer *l, int inputs);

#ifdef GPU
void forward_cost_layer_gpu(layer l, network_state state);
void backward_cost_layer_gpu(const layer l, network_state state);
#endif

#endif
