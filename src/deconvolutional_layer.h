#ifndef DECONVOLUTIONAL_LAYER_H
#define DECONVOLUTIONAL_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

#ifdef GPU
void forward_deconvolutional_layer_gpu(layer_t layer, network_state state);
void backward_deconvolutional_layer_gpu(layer_t layer, network_state state);
void update_deconvolutional_layer_gpu(layer_t layer, float learning_rate, float momentum, float decay);
void push_deconvolutional_layer(layer_t layer);
void pull_deconvolutional_layer(layer_t layer);
#endif

layer_t make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, ACTIVATION activation);
void resize_deconvolutional_layer(layer_t *layer, int h, int w);
void forward_deconvolutional_layer(const layer_t layer, network_state state);
void update_deconvolutional_layer(layer_t layer, float learning_rate, float momentum, float decay);
void backward_deconvolutional_layer(layer_t layer, network_state state);

image get_deconvolutional_image(layer_t layer);
image get_deconvolutional_delta(layer_t layer);
image get_deconvolutional_filter(layer_t layer, int i);

int deconvolutional_out_height(layer_t layer);
int deconvolutional_out_width(layer_t layer);

#endif

