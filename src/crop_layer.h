#ifndef CROP_LAYER_H
#define CROP_LAYER_H

#include "image.h"
#include "layer.h"
#include "network.h"

image get_crop_image(layer_t l);
layer_t make_crop_layer(int batch, int h, int w, int c, int crop_height, int crop_width, int flip, float angle, float saturation, float exposure);
void forward_crop_layer(const layer_t l, network_state state);
void resize_crop_layer(layer_t *l, int w, int h);

#ifdef GPU
void forward_crop_layer_gpu(layer_t l, network_state state);
#endif

#endif

