#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

#ifdef GPU
void forward_convolutional_layer_gpu(layer_t layer, network_state state);
void backward_convolutional_layer_gpu(layer_t layer, network_state state);
void update_convolutional_layer_gpu(layer_t layer, int batch, float learning_rate, float momentum, float decay);

void push_convolutional_layer(layer_t layer);
void pull_convolutional_layer(layer_t layer);

void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size);
#endif

layer_t make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int pad, ACTIVATION activation, int batch_normalization, int binary);
void denormalize_convolutional_layer(layer_t l);
void resize_convolutional_layer(layer_t *layer, int w, int h);
void forward_convolutional_layer(const layer_t layer, network_state state);
void update_convolutional_layer(layer_t layer, int batch, float learning_rate, float momentum, float decay);
image *visualize_convolutional_layer(layer_t layer, char *window, image *prev_filters);
void binarize_filters(float *filters, int n, int size, float *binary);
void swap_binary(layer_t *l);
void binarize_filters2(float *filters, int n, int size, char *binary, float *scales);

void backward_convolutional_layer(layer_t layer, network_state state);

void add_bias(float *output, float *biases, int batch, int n, int size);
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size);

image get_convolutional_image(layer_t layer);
image get_convolutional_delta(layer_t layer);
image get_convolutional_filter(layer_t layer, int i);

int convolutional_out_height(layer_t layer);
int convolutional_out_width(layer_t layer);
void rescale_filters(layer_t l, float scale, float trans);
void rgbgr_filters(layer_t l);

#endif

