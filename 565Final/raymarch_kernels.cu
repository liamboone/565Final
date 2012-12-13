#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cutil_math.h>
#include "cutil_inline.h"

#define DEG_TO_RAD 0.017453292519943 

__global__
void render_k( float4 * out, float3 campos, unsigned int width, unsigned int height )
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if( x >= width || y >= height )
	{
		return;
	}

	out[x+y*width] = make_float4( x / (float)width, y / (float)width, 0.0, 1.0 );
}


extern "C" 
void render( dim3 dimGrid, dim3 dimBlock, float4 * out, float3 campos, unsigned int width, unsigned int height )
{
	float4 * tmp;
	cudaMalloc( &tmp, width * height * sizeof( float4 ) );
	render_k<<< dimGrid, dimBlock >>>( tmp, campos, width, height );
	cudaMemcpy( out, tmp, width * height * sizeof( float4 ), cudaMemcpyDeviceToHost );
	cudaFree( tmp );
}
