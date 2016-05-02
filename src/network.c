#include <stdio.h>
#include <time.h>
#include <stdbool.h>
#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "blas.h"

#include "crop_layer.h"
#include "connected_layer.h"
#include "rnn_layer.h"
#include "crnn_layer.h"
#include "local_layer.h"
#include "convolutional_layer.h"
#include "activation_layer.h"
#include "deconvolutional_layer.h"
#include "detection_layer.h"
#include "normalization_layer.h"
#include "maxpool_layer.h"
#include "avgpool_layer.h"
#include "cost_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "route_layer.h"
#include "shortcut_layer.h"

int get_current_batch(network net)
{
    int batch_num = (*net.seen)/(net.batch*net.subdivisions);
    return batch_num;
}

void reset_momentum(network net)
{
    if (net.momentum == 0) return;
    net.learning_rate = 0;
    net.momentum = 0;
    net.decay = 0;
    #ifdef GPU
        if(gpu_index >= 0) update_network_gpu(net);
    #endif
}

float get_current_rate(network net)
{
    int batch_num = get_current_batch(net);
    int i;
    float rate;
    switch (net.policy) {
        case CONSTANT:
            return net.learning_rate;
        case STEP:
            return net.learning_rate * pow(net.scale, batch_num/net.step);
        case STEPS:
            rate = net.learning_rate;
            for(i = 0; i < net.num_steps; ++i){
                if(net.steps[i] > batch_num) return rate;
                rate *= net.scales[i];
                if(net.steps[i] > batch_num - 1) reset_momentum(net);
            }
            return rate;
        case EXP:
            return net.learning_rate * pow(net.gamma, batch_num);
        case POLY:
            return net.learning_rate * pow(1 - (float)batch_num / net.max_batches, net.power);
        case SIG:
            return net.learning_rate * (1./(1.+exp(net.gamma*(batch_num - net.step))));
        default:
            fprintf(stderr, "Policy is weird!\n");
            return net.learning_rate;
    }
}

char *get_layer_string(LAYER_TYPE a)
{
    switch(a){
        case CONVOLUTIONAL:
            return "convolutional";
        case ACTIVE:
            return "activation";
        case LOCAL:
            return "local";
        case DECONVOLUTIONAL:
            return "deconvolutional";
        case CONNECTED:
            return "connected";
        case RNN:
            return "rnn";
        case CRNN:
            return "crnn";
        case MAXPOOL:
            return "maxpool";
        case AVGPOOL:
            return "avgpool";
        case SOFTMAX:
            return "softmax";
        case DETECTION:
            return "detection";
        case DROPOUT:
            return "dropout";
        case CROP:
            return "crop";
        case COST:
            return "cost";
        case ROUTE:
            return "route";
        case SHORTCUT:
            return "shortcut";
        case NORMALIZATION:
            return "normalization";
        default:
            break;
    }
    return "none";
}

network make_network(int n)
{
    network net = {0};
    net.n = n;
    net.layers = calloc(net.n, sizeof(layer_t));
    net.seen = calloc(1, sizeof(int));
    #ifdef GPU
    net.input_gpu = calloc(1, sizeof(float *));
    net.truth_gpu = calloc(1, sizeof(float *));
    #endif
    return net;
}

void forward_network(network net, network_state state)
{
	for (int i = 0; i < net.n; ++i) {
		state.index = i;
		layer_t l = net.layers[i];
		if (l.delta) {
			scal_cpu(l.outputs * l.batch, 0, l.delta, 1);
		}
		switch (l.type) {
		case CONVOLUTIONAL:
			forward_convolutional_layer(l, state);
			break;
		case DECONVOLUTIONAL:
			forward_deconvolutional_layer(l, state);
			break;
		case ACTIVE:
			forward_activation_layer(l, state);
			break;
		case LOCAL:
			forward_local_layer(l, state);
			break;
		case NORMALIZATION:
			forward_normalization_layer(l, state);
			break;
		case DETECTION:
			forward_detection_layer(l, state);
			break;
		case CONNECTED:
			forward_connected_layer(l, state);
			break;
		case RNN:
			forward_rnn_layer(l, state);
			break;
		case CRNN:
			forward_crnn_layer(l, state);
			break;
		case CROP:
			forward_crop_layer(l, state);
			break;
		case COST:
			forward_cost_layer(l, state);
			break;
		case SOFTMAX:
			forward_softmax_layer(l, state);
			break;
		case MAXPOOL:
			forward_maxpool_layer(l, state);
			break;
		case AVGPOOL:
			forward_avgpool_layer(l, state);
			break;
		case DROPOUT:
			forward_dropout_layer(l, state);
			break;
		case ROUTE:
			forward_route_layer(l, net);
			break;
		case SHORTCUT:
			forward_shortcut_layer(l, state);
		default:
			break;
		}
		state.input = l.output;
	}
}

static void update_network(network net)
{
	int update_batch = net.batch * net.subdivisions;
	float rate = get_current_rate(net);
	for (int i = 0; i < net.n; ++i) {
		layer_t l = net.layers[i];
		switch (l.type) {
		case CONVOLUTIONAL:
			update_convolutional_layer(l, update_batch, rate, net.momentum, net.decay);
			break;
		case DECONVOLUTIONAL:
			update_deconvolutional_layer(l, rate, net.momentum, net.decay);
			break;
		case CONNECTED:
			update_connected_layer(l, update_batch, rate, net.momentum, net.decay);
			break;
		case RNN:
			update_rnn_layer(l, update_batch, rate, net.momentum, net.decay);
			break;
		case CRNN:
			update_crnn_layer(l, update_batch, rate, net.momentum, net.decay);
			break;
		case LOCAL:
			update_local_layer(l, update_batch, rate, net.momentum, net.decay);
		default:
			break;
		}
	}
}

float *get_network_output(network net)
{
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return net.layers[i].output;
}

static float get_network_cost(network net)
{
	float sum = 0;
	int count = 0;
	for (int i = 0; i < net.n; ++i) {
		switch (net.layers[i].type) {
		case COST:
		case DETECTION:
			sum += net.layers[i].cost[0];
			++count;
		default:
			break;
		}
	}
	return sum / count; // TODO: watch out for div-by-zero
}

int get_predicted_class_network(network net)
{
    float *out = get_network_output(net);
    int k = get_network_output_size(net);
    return max_index(out, k);
}

void backward_network(network net, network_state state)
{
	float *original_input = state.input;
	float *original_delta = state.delta;
	for (int i = net.n-1; i >= 0; --i) {
		state.index = i;
		if (i == 0) {
			state.input = original_input;
			state.delta = original_delta;
		} else {
			layer_t prev = net.layers[i-1];
			state.input = prev.output;
			state.delta = prev.delta;
		}
		layer_t l = net.layers[i];
		switch (l.type) {
		case CONVOLUTIONAL:
			backward_convolutional_layer(l, state);
			break;
		case DECONVOLUTIONAL:
			backward_deconvolutional_layer(l, state);
			break;
		case ACTIVE:
			backward_activation_layer(l, state);
			break;
		case NORMALIZATION:
			backward_normalization_layer(l, state);
			break;
		case MAXPOOL:
			backward_maxpool_layer(l, state);
			break;
		case AVGPOOL:
			backward_avgpool_layer(l, state);
			break;
		case DROPOUT:
			backward_dropout_layer(l, state);
			break;
		case DETECTION:
			backward_detection_layer(l, state);
			break;
		case SOFTMAX:
			backward_softmax_layer(l, state);
			break;
		case CONNECTED:
			backward_connected_layer(l, state);
			break;
		case RNN:
			backward_rnn_layer(l, state);
			break;
		case CRNN:
			backward_crnn_layer(l, state);
			break;
		case LOCAL:
			backward_local_layer(l, state);
			break;
		case COST:
			backward_cost_layer(l, state);
			break;
		case ROUTE:
			backward_route_layer(l, net);
			break;
		case SHORTCUT:
			backward_shortcut_layer(l, state);
		default:
			break;
		}
	}
}

float train_network_datum(network net, float *x, float *y)
{
    *net.seen += net.batch;
#ifdef GPU
    if(gpu_index >= 0) return train_network_datum_gpu(net, x, y);
#endif
    network_state state;
    state.index = 0;
    state.net = net;
    state.input = x;
    state.delta = NULL;
    state.truth = y;
    state.train = true;
    forward_network(net, state);
    backward_network(net, state);
    float error = get_network_cost(net);
    if(((*net.seen)/net.batch)%net.subdivisions == 0) update_network(net);
    return error;
}

float train_network_sgd(network net, data d, int n)
{
    int batch = net.batch;
    float *X = calloc(batch*d.X.cols, sizeof(float));
    float *y = calloc(batch*d.y.cols, sizeof(float));

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_random_batch(d, batch, X, y);
        float err = train_network_datum(net, X, y);
        sum += err;
    }
    free(X);
    free(y);
    return (float)sum/(n*batch);
}

float train_network(network net, data d)
{
    int batch = net.batch;
    int n = d.X.rows / batch;
    float *X = calloc(batch*d.X.cols, sizeof(float));
    float *y = calloc(batch*d.y.cols, sizeof(float));

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, X, y);
        float err = train_network_datum(net, X, y);
        sum += err;
    }
    free(X);
    free(y);
    return (float)sum/(n*batch);
}

float train_network_batch(network net, data d, int n)
{
    int i,j;
    network_state state;
    state.index = 0;
    state.net = net;
    state.train = true;
    state.delta = NULL;
    float sum = 0;
    int batch = 2;
    for(i = 0; i < n; ++i){
        for(j = 0; j < batch; ++j){
            int index = rand()%d.X.rows;
            state.input = d.X.vals[index];
            state.truth = d.y.vals[index];
            forward_network(net, state);
            backward_network(net, state);
            sum += get_network_cost(net);
        }
        update_network(net);
    }
    return (float)sum/(n*batch);
}

void set_batch_network(network *net, int b)
{
    net->batch = b;
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].batch = b;
    }
}

int resize_network(network *net, int w, int h)
{
	//if(w == net->w && h == net->h) return 0;
	net->w = w;
	net->h = h;
	int inputs = 0;
	//fprintf(stderr, "Resizing to %d x %d...", w, h);
	//fflush(stderr);
	for (int i = 0; i < net->n; ++i) {
		layer_t l = net->layers[i];
		switch (l.type) {
		case CONVOLUTIONAL:
			resize_convolutional_layer(&l, w, h);
			break;
		case CROP:
			resize_crop_layer(&l, w, h);
			break;
		case MAXPOOL:
			resize_maxpool_layer(&l, w, h);
			break;
		case AVGPOOL:
			resize_avgpool_layer(&l, w, h);
			break;
		case NORMALIZATION:
			resize_normalization_layer(&l, w, h);
			break;
		case COST:
			resize_cost_layer(&l, inputs);
			break;
		default:
			error("Cannot resize this type of layer");
			break;
		}
		inputs = l.outputs;
		net->layers[i] = l;
		w = l.out_w;
		h = l.out_h;
		if (l.type == AVGPOOL) {
			break;
		}
	}
	//fprintf(stderr, " Done!\n");
	return 0;
}

int get_network_output_size(network net)
{
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return net.layers[i].outputs;
}

int get_network_input_size(network net)
{
    return net.layers[0].inputs;
}

layer_t get_network_detection_layer(network net)
{
    int i;
    for(i = 0; i < net.n; ++i){
        if(net.layers[i].type == DETECTION){
            return net.layers[i];
        }
    }
    fprintf(stderr, "Detection layer not found!!\n");
    layer_t l = {0};
    return l;
}

image get_network_image_layer(network net, int i)
{
    layer_t l = net.layers[i];
    if (l.out_w && l.out_h && l.out_c){
        return float_to_image(l.out_w, l.out_h, l.out_c, l.output);
    }
    image def = {0};
    return def;
}

image get_network_image(network net)
{
    int i;
    for(i = net.n-1; i >= 0; --i){
        image m = get_network_image_layer(net, i);
        if(m.h != 0) return m;
    }
    image def = {0};
    return def;
}

void visualize_network(network net)
{
    image *prev = 0;
    int i;
    char buff[256];
    for(i = 0; i < net.n; ++i){
        sprintf(buff, "Layer %d", i);
        layer_t l = net.layers[i];
        if(l.type == CONVOLUTIONAL){
            prev = visualize_convolutional_layer(l, buff, prev);
        }
    } 
}

void top_predictions(network net, int k, int *index)
{
    int size = get_network_output_size(net);
    float *out = get_network_output(net);
    top_k(out, size, k, index);
}


float *network_predict(network net, float *input)
{
#ifdef GPU
    if(gpu_index >= 0)  return network_predict_gpu(net, input);
#endif

    network_state state;
    state.net = net;
    state.index = 0;
    state.input = input;
    state.truth = NULL;
    state.train = false;
    state.delta = NULL;
    forward_network(net, state);
    float *out = get_network_output(net);
    return out;
}

matrix network_predict_data_multi(network net, data test, int n)
{
    int i,j,b,m;
    int k = get_network_output_size(net);
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net.batch*test.X.rows, sizeof(float));
    for(i = 0; i < test.X.rows; i += net.batch){
        for(b = 0; b < net.batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        for(m = 0; m < n; ++m){
            float *out = network_predict(net, X);
            for(b = 0; b < net.batch; ++b){
                if(i+b == test.X.rows) break;
                for(j = 0; j < k; ++j){
                    pred.vals[i+b][j] += out[j+b*k]/n;
                }
            }
        }
    }
    free(X);
    return pred;   
}

matrix network_predict_data(network net, data test)
{
    int i,j,b;
    int k = get_network_output_size(net);
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net.batch*test.X.cols, sizeof(float));
    for(i = 0; i < test.X.rows; i += net.batch){
        for(b = 0; b < net.batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        float *out = network_predict(net, X);
        for(b = 0; b < net.batch; ++b){
            if(i+b == test.X.rows) break;
            for(j = 0; j < k; ++j){
                pred.vals[i+b][j] = out[j+b*k];
            }
        }
    }
    free(X);
    return pred;   
}

void print_network(network net)
{
    int i,j;
    for(i = 0; i < net.n; ++i){
        layer_t l = net.layers[i];
        float *output = l.output;
        int n = l.outputs;
        float mean = mean_array(output, n);
        float vari = variance_array(output, n);
        fprintf(stderr, "Layer %d - Mean: %f, Variance: %f\n",i,mean, vari);
        if(n > 100) n = 100;
        for(j = 0; j < n; ++j) fprintf(stderr, "%f, ", output[j]);
        if(n == 100)fprintf(stderr,".....\n");
        fprintf(stderr, "\n");
    }
}

void compare_networks(network n1, network n2, data test)
{
    matrix g1 = network_predict_data(n1, test);
    matrix g2 = network_predict_data(n2, test);
    int i;
    int a,b,c,d;
    a = b = c = d = 0;
    for(i = 0; i < g1.rows; ++i){
        int truth = max_index(test.y.vals[i], test.y.cols);
        int p1 = max_index(g1.vals[i], g1.cols);
        int p2 = max_index(g2.vals[i], g2.cols);
        if(p1 == truth){
            if(p2 == truth) ++d;
            else ++c;
        }else{
            if(p2 == truth) ++b;
            else ++a;
        }
    }
    printf("%5d %5d\n%5d %5d\n", a, b, c, d);
    float num = pow((abs(b - c) - 1.), 2.);
    float den = b + c;
    printf("%f\n", num/den); 
}

float network_accuracy(network net, data d)
{
    matrix guess = network_predict_data(net, d);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

float *network_accuracies(network net, data d, int n)
{
    static float acc[2];
    matrix guess = network_predict_data(net, d);
    acc[0] = matrix_topk_accuracy(d.y, guess, 1);
    acc[1] = matrix_topk_accuracy(d.y, guess, n);
    free_matrix(guess);
    return acc;
}


float network_accuracy_multi(network net, data d, int n)
{
    matrix guess = network_predict_data_multi(net, d, n);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

void free_network(network net)
{
    int i;
    for(i = 0; i < net.n; ++i){
        free_layer(net.layers[i]);
    }
    free(net.layers);
    #ifdef GPU
    if(*net.input_gpu) cuda_free(*net.input_gpu);
    if(*net.truth_gpu) cuda_free(*net.truth_gpu);
    if(net.input_gpu) free(net.input_gpu);
    if(net.truth_gpu) free(net.truth_gpu);
    #endif
}
