#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cutil_math.h>
#include "cutil_inline.h"

__global__
void process_depth_k( float * dRGBA, unsigned short * depthRAW, unsigned int width, unsigned int height  )
{

}

extern "C" void process_depth( dim3 dimGrid, dim3 dimBlock, float * dRGBA, unsigned short * depthRAW, unsigned int width, unsigned int height )
{
	process_depth_k <<< dimGrid, dimBlock >>> ( dRGBA, depthRAW, width, height );
	cutilCheckMsg("getGrid");
}