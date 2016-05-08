#include "activation_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

layer_t make_activation_layer(int batch, int inputs, ACTIVATION activation)
{
    layer_t l = {0};
    l.type = ACTIVE;

    l.inputs = inputs;
    l.outputs = inputs;
    l.batch=batch;

    l.output = calloc(batch*inputs, sizeof(float*));
    l.delta = calloc(batch*inputs, sizeof(float*));

#ifdef GPU
    l.output_gpu = cuda_make_array(l.output, inputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
#endif
    l.activation = activation;
    fprintf(stderr, "Activation Layer: %d inputs\n", inputs);
    return l;
}

void forward_activation_layer(layer_t l, network_state state)
{
    fltcpy(l.output, state.input, l.outputs * l.batch);
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_activation_layer(layer_t l, network_state state)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    fltcpy(state.delta, l.delta, l.outputs * l.batch);
}

#ifdef GPU

void forward_activation_layer_gpu(layer_t l, network_state state)
{
    copy_ongpu(l.outputs*l.batch, state.input, 1, l.output_gpu, 1);
    activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_activation_layer_gpu(layer_t l, network_state state)
{
    gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    copy_ongpu(l.outputs*l.batch, l.delta_gpu, 1, state.delta, 1);
}
#endif
