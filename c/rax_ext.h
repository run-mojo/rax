//
//
//

#ifndef RAX_RAX_EXT_H
#define RAX_RAX_EXT_H

#include "rax.h"

extern char *RAX_GREATER;
extern char *RAX_GREATER_EQUAL;
extern char *RAX_LESSER;
extern char *RAX_LESSER_EQUAL;
extern char *RAX_EQUAL;
extern char *RAX_MIN;
extern char *RAX_MAX;

raxIterator *raxIteratorNew(rax *rt);
void raxIteratorFree(raxIterator *it);
void *raxIteratorData(raxIterator *it);

int raxIteratorSize() {
    return sizeof(raxIterator);
}

#endif //RAX_RAX_EXT_H
