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

// init with libc malloc
void* (*rax_malloc)(size_t) = malloc;
// init with libc realloc
void* (*rax_realloc)(void*,size_t) = realloc;
// init with libc free
void (*rax_free)(void*) = free;

raxIterator *raxIteratorNew(rax *rt) {
    // Allocate on the heap.
    raxIterator *it = rax_malloc(sizeof(raxIterator));
    raxStart(it, rt);
    return it;
}
