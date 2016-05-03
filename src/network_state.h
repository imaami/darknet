#ifndef NETWORK_STATE_H
#define NETWORK_STATE_H

#include <stddef.h>
#include <stdbool.h>

struct network;
typedef struct network_state network_state;

struct network_state {
	struct network *net;
	size_t          index;
	bool            train;
	float          *delta;
	float          *input;
	float          *truth;
};

#define NETWORK_STATE_INITIALIZER { NULL, 0, false, NULL, NULL, NULL }
#define NETWORK_STATE(x) network_state x = NETWORK_STATE_INITIALIZER

extern void
network_state_init (network_state  *state,
                    struct network *net,
                    size_t          index,
                    bool            train,
                    float          *delta,
                    float          *input,
                    float          *truth);

#endif // NETWORK_STATE_H
