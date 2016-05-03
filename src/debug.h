#ifndef DEBUG_H
#define DEBUG_H

#include <stdio.h>

#define DBG(fmt, args...) do{fprintf(stderr,"%s: "fmt"\n",__func__ , ##args);}while(0)

#endif // DEBUG_H
