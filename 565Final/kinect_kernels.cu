#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cutil_math.h>
#include "cutil_inline.h"

#define DEG_TO_RAD 0.017453292519943 

float4 * dRGBA;
unsigned short * dRAW;
int * dWARP;

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

__device__
float3 getWorldSpaceFromDepthSpace( int x, int y, short depth, int width, int height, float2 XYScale )
{
	float3 XYZ;
	float phi   = ( (float)x / (float)width  * 57.0f - 28.5f ) * DEG_TO_RAD;
	float theta = ( (float)y / (float)height * 43.0f + 68.5f ) * DEG_TO_RAD;
	XYZ.x = depth;
	XYZ.y = (int)(depth * tan(phi));
	XYZ.z = (int)(depth / tan(theta) / cos(phi));
	return XYZ;
}

__device__
int getIndex( int x, int y, int width, int limx, int limy )
{
	if (x < 0 || y < 0 || x >= limx || y >= limy)
		return -1;
	return width * y + x;
}

__global__
void clear_k( int * warp, unsigned int width, unsigned int height )
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if( x == width-1 || y == height-1 )
	{
		return;
	}

	unsigned int cindex = (height-y-1)*width+x;
	warp[cindex] = 3000;
}

__global__
void make_pretty_k( float4 * color, int * warp, unsigned int width, unsigned int height )
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if( x == width-1 || y == height-1 )
	{
		return;
	}

	unsigned int cindex = (height-y-1)*width+x;
	float fdepth = warp[cindex] == 0 ? 0.0 : ( 1.0 - ( warp[cindex] - 400.0 ) / 3000.0 );
	
	color[cindex] = make_float4( fdepth );
}

__global__
void process_depth_k( int * warp, unsigned short * raw, unsigned int width, unsigned int height, float2 XYScale, int target )
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	int rHeight = 315, rWidth = 434;

	if( x == width-1 || y == height-1 )
	{
		return;
	}
	
	unsigned int index = y*width+x;
	unsigned int cindex = (height-y-1)*width+x;

	unsigned short player = getPlayerFromRAW( raw[index] );
	unsigned short depth = getDepthFromRAW( raw[index] );

	float3 worldpos = getWorldSpaceFromDepthSpace( x, y, depth, width, height, XYScale );

	int wpx = (worldpos.y / 10 ) + rWidth / 2.0;
	int wpy = (worldpos.z / 10) + rHeight / 2.0;
		
	int warpindex = getIndex( wpx, wpy, width, rWidth, rHeight );

	if( warpindex >= 0 && target == player )
		atomicMin( &(warp[warpindex]), (int)depth );
	else
		atomicMin( &(warp[cindex]), (int)4000 );
}

extern "C" void process_depth( dim3 dimGrid, dim3 dimBlock, float4 * depthRGBA, unsigned short * depthRAW, unsigned int width, unsigned int height, float xyscale, int target )
{
	cudaMemcpy( dRAW, depthRAW,  width * height * sizeof( unsigned short ), cudaMemcpyHostToDevice ); cutilCheckMsg("RAW Depth Transfer");
	clear_k<<< dimGrid, dimBlock >>> ( dWARP, width, height );
	process_depth_k <<< dimGrid, dimBlock >>> ( dWARP, dRAW, width, height, make_float2( xyscale ), target ); cutilCheckMsg("Depth Process");
	make_pretty_k <<< dimGrid, dimBlock >>> ( dRGBA, dWARP, width, height ); cutilCheckMsg("Convert to Image");
	cudaMemcpy( depthRGBA, dRGBA, width * height * sizeof( float4 ), cudaMemcpyDeviceToHost ); cutilCheckMsg("RGBA Depth Transfer");
}

extern "C" void cudaInit( unsigned int width, unsigned int height )
{
	cutilCheckMsg("Before");
	cudaMalloc( &dRGBA, width * height * sizeof( float4 ) );
	cutilCheckMsg("CUDA Malloc");
	cudaMalloc( &dRAW, width * height * sizeof( unsigned short ) );
	cutilCheckMsg("CUDA Malloc");
	cudaMalloc( &dWARP, width * height * sizeof( int ) );
	cutilCheckMsg("CUDA Malloc");
}