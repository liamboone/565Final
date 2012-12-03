#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cutil_math.h>
#include "cutil_inline.h"

float4 * dRGBA;
unsigned short * dRAW;

__device__
unsigned short getDepthFromRAW( unsigned short raw )
{
	return ( raw >> 3 );
}

__device__
unsigned short getPlayerFromRAW( unsigned short raw )
{
	return ( raw & 0x7 );
}

__global__
void process_depth_k( float4 * color, unsigned short * raw, unsigned int width, unsigned int height  )
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if( x == width-1 || y == height-1 )
	{
		return;
	}
	
	unsigned int index = y*width+x;
	unsigned int cindex = (height-y-1)*width+x;

	unsigned short player = getPlayerFromRAW( raw[index] );
	unsigned short depth = getDepthFromRAW( raw[index] );

	if( player > 0 )
	{
		color[cindex].x = ( player & 0x1 );
		color[cindex].y = ( player & 0x2 ) >> 1;
		color[cindex].z = ( player & 0x4 ) >> 2;
	}
	else
	{
		float fdepth = depth == 0 ? 0.0 : ( 1.0 - ( depth - 400.0 ) / 3000.0 );
		color[cindex] = make_float4( fdepth );
	}
}

extern "C" void process_depth( dim3 dimGrid, dim3 dimBlock, float4 * depthRGBA, unsigned short * depthRAW, unsigned int width, unsigned int height )
{
	cudaMemcpy( dRAW, depthRAW,  width * height * sizeof( unsigned short ), cudaMemcpyHostToDevice ); cutilCheckMsg("RAW Depth Transfer");
	process_depth_k <<< dimGrid, dimBlock >>> ( dRGBA, dRAW, width, height ); cutilCheckMsg("Depth Process");
	cudaMemcpy( depthRGBA, dRGBA, width * height * sizeof( float4 ), cudaMemcpyDeviceToHost ); cutilCheckMsg("RGBA Depth Transfer");
}

extern "C" void cudaInit()
{
	cutilCheckMsg("Before");
	cudaMalloc( &dRGBA, 320 * 240 * sizeof( float4 ) );
	cutilCheckMsg("CUDA Malloc");
	cudaMalloc( &dRAW, 320 * 240 * sizeof( unsigned short ) );
	cutilCheckMsg("CUDA Malloc");
}