#ifndef DARKNET_CFG_H
#define DARKNET_CFG_H

#include <string.h>

typedef enum {
	CFG_SECTION_TYPE_NONE            = 0x00,
	CFG_SECTION_TYPE_NETWORK         = 0x01,
	CFG_SECTION_TYPE_CONVOLUTIONAL   = 0x02,
	CFG_SECTION_TYPE_DECONVOLUTIONAL = 0x03,
	CFG_SECTION_TYPE_CONNECTED       = 0x04,
	CFG_SECTION_TYPE_MAXPOOL         = 0x05,
	CFG_SECTION_TYPE_SOFTMAX         = 0x06,
	CFG_SECTION_TYPE_DETECTION       = 0x07,
	CFG_SECTION_TYPE_DROPOUT         = 0x08,
	CFG_SECTION_TYPE_CROP            = 0x09,
	CFG_SECTION_TYPE_ROUTE           = 0x0a,
	CFG_SECTION_TYPE_COST            = 0x0b,
	CFG_SECTION_TYPE_NORMALIZATION   = 0x0c,
	CFG_SECTION_TYPE_AVGPOOL         = 0x0d,
	CFG_SECTION_TYPE_LOCAL           = 0x0e,
	CFG_SECTION_TYPE_SHORTCUT        = 0x0f,
	CFG_SECTION_TYPE_ACTIVE          = 0x10,
	CFG_SECTION_TYPE_RNN             = 0x11,
	CFG_SECTION_TYPE_CRNN            = 0x12
} cfg_section_type_t;

__attribute__((always_inline))
static inline cfg_section_type_t
cfg_get_section_type(char *s)
{
	if (s[0] != '[') {
		return CFG_SECTION_TYPE_NONE;
	}

	switch (s[1]) {
	case 'a':
		switch (s[2]) {
		case 'c':
			if (!strncmp(s + 3, "tive]", 5)) {
				return CFG_SECTION_TYPE_ACTIVE;
			}
			break;

		case 'v':
			if (s[3] == 'g' &&
			    (s[4] == ']' || !strncmp(s + 4, "pool]", 5))) {
				return CFG_SECTION_TYPE_AVGPOOL;
			}
		}
		break;

	case 'c':
		switch (s[2]) {
		case 'o':
			switch (s[3]) {
			case 'n':
				switch (s[4]) {
				case 'n':
					if (s[5] == ']' ||
					    !strncmp(s + 5, "ected]", 6)) {
						return CFG_SECTION_TYPE_CONNECTED;
					}
					break;

				case 'v':
					if (s[5] == ']' ||
					    !strncmp(s + 5, "olutional]", 10)) {
						return CFG_SECTION_TYPE_CONVOLUTIONAL;
					}
				}
				break;

			case 's':
				if (s[4] == 't' && s[5] == ']') {
					return CFG_SECTION_TYPE_COST;
				}
			}
			break;

		case 'r':
				if (s[3] == 'n' && s[4] == 'n' && s[5] == ']') {
					return CFG_SECTION_TYPE_CRNN;
				}
				if (s[3] == 'o' && s[4] == 'p' && s[5] == ']') {
					return CFG_SECTION_TYPE_CROP;
				}
		}
		break;

	case 'd':
		switch (s[2]) {
		case 'e':
			if (!strncmp(s + 3, "conv", 4) &&
			    (s[7] == ']' ||
			     !strncmp(s + 7, "olutional]", 10))) {
				return CFG_SECTION_TYPE_DECONVOLUTIONAL;
			}
			if (!strncmp(s + 3, "tection]", 8)) {
				return CFG_SECTION_TYPE_DETECTION;
			}
			break;

		case 'r':
			if (!strncmp(s + 3, "opout]", 6)) {
				return CFG_SECTION_TYPE_DROPOUT;
			}
		}
		break;

	case 'l':
		switch (s[2]) {
		case 'o':
			if (!strncmp(s + 3, "cal]", 4)) {
				return CFG_SECTION_TYPE_LOCAL;
			}
			break;

		case 'r':
			if (s[3] == 'n' && s[4] == ']') {
				return CFG_SECTION_TYPE_NORMALIZATION;
			}
		}
		break;

	case 'm':
		if (s[2] == 'a' && s[3] == 'x' &&
		    (s[4] == ']' || !strncmp(s + 4, "pool]", 5))) {
			return CFG_SECTION_TYPE_MAXPOOL;
		}
		break;

	case 'n':
		// TODO: network, normalization
		break;

	case 'r':
		// TODO: rnn, route
		break;

	case 's':
		// TODO: shortcut, softmax

	default:
		break;
	}

	return CFG_SECTION_TYPE_NONE;
}

#endif // DARKNET_CFG_H
