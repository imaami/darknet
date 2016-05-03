#include "network_state.h"

#include <stddef.h>
#include <stdbool.h>

void
network_state_init (network_state  *state,
                    struct network *net,
                    size_t          index,
                    bool            train,
                    float          *delta,
                    float          *input,
                    float          *truth)
{
	state->truth = truth;
	state->input = input;
	state->delta = delta;
	state->train = train;
	state->index = index;
	state->net = net;
}
