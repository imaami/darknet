#ifndef NORMALIZATION_LAYER_H
#define NORMALIZATION_LAYER_H

#include "image.h"
#include "layer.h"
#include "network.h"

layer_t make_normalization_layer(int batch, int w, int h, int c, int size, float alpha, float beta, float kappa);
void resize_normalization_layer(layer_t *layer, int h, int w);
void forward_normalization_layer(const layer_t layer, network_state state);
void backward_normalization_layer(const layer_t layer, network_state state);
void visualize_normalization_layer(layer_t layer, char *window);

#ifdef GPU
void forward_normalization_layer_gpu(const layer_t layer, network_state state);
void backward_normalization_layer_gpu(const layer_t layer, network_state state);
#endif

#endif
