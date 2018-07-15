//
//
//

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <errno.h>
#include <math.h>
#include "rax_ext.h"
#include "rax_malloc.h"

char *RAX_GREATER = ">";
char *RAX_GREATER_EQUAL = ">=";
char *RAX_LESSER = "<";
char *RAX_LESSER_EQUAL = "<=";
char *RAX_EQUAL = "=";
char *RAX_MIN = "^";
char *RAX_MAX = "$";

raxIterator *raxIteratorNew(rax *rt) {
    // Allocate on the heap.
    raxIterator *it = rax_malloc(sizeof(raxIterator));
    raxStart(it, rt);
    return it;
}

void raxIteratorFree(raxIterator *it) {
    //
    rax_free(it);
}

#include "rax_ext.h"
