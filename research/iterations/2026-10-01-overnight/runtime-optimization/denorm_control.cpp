#include <xmmintrin.h>
extern "C" unsigned get_mxcsr() {return _mm_getcsr();}
extern "C" void set_mxcsr(unsigned value) {_mm_setcsr(value);}
