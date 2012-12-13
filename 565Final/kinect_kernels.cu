#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cutil_math.h>
#include "cutil_inline.h"

#define DEG_TO_RAD 0.017453292519943 

float4 * dRGBA;
unsigned short * dRAW;
int * dWARP;
int * dBUFF;
int * dHUFF;
int * dEDGE;
int * hmax;

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
void clear_k( int * warp, int * hough, int * edge, int * maximum, unsigned int width, unsigned int height )
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if( x >= width || y >= height )
	{
		return;
	}

	unsigned int cindex = (height-y-1)*width+x;
	if( cindex == 0 ) *maximum = 0;
	warp[cindex] = 3000;
	hough[cindex] = 0;
	edge[cindex] = 0;
}

__global__
void make_pretty_k( float4 * color, int * warp, int * edge, int* hough, int * maximum, unsigned int width, unsigned int height )
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if( x == width-1 || y == height-1 || x == 0 || y == 0 )
	{
		return;
	}

	unsigned int cindex = (height-y-1)*width+x;
	float fdepth = warp[cindex] == 0 ? 0.0 : ( 1.0 - ( warp[cindex] - 400.0 ) / 2600.0 );
	
	color[cindex] = make_float4(	hough[cindex] >= *maximum, 
									edge[cindex] > 0, 
									0.0,//( fdepth > 0 ? sqrt( (float)min( hough[cindex] / 40.0, 1.0 ) ) : 0.0 ), 
									1.0 );
}

__global__
void max_5x5_k( int * buff, int * warp, unsigned int width, unsigned int height )
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if( x == width-1 || y == height-1 || x == 0 || y == 0 )
	{
		return;
	}

	int index = (height-y-1)*width+x;

	int maximum = warp[index];
	for( int i = -2; i <= 2; i ++ )
	{
		for( int j = -2; j <= 2; j ++ )
		{
			maximum = max( maximum, warp[index+i+j*width] );
		}
	}
	buff[index] = maximum;
}


__global__
void find_head_k( int * maximum, int * warp, int * hough, unsigned int width, unsigned int height )
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if( x == width-1 || y == height-1 || x == 0 || y == 0 )
	{
		return;
	}

	int index = (height-y-1)*width+x;

	if( warp[index] < 3000 ) atomicMax( maximum, hough[index] );
}

__global__
void hough_k( int * buff, int * edge, unsigned int width, unsigned int height )
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if( x == width-1 || y == height-1 || x == 0 || y == 0 )
	{
		return;
	}

	int index = (height-y-1)*width+x;
	
	for( int t = 0; t < 180; t += 10 )
	{
		int xoff = 10.0 * cos( t * DEG_TO_RAD ) + 0.5;
		int yoff = 10.0 * sin( t * DEG_TO_RAD ) + 0.5;

		for( int i = -1; i <= 1; i ++ )
		{
			for( int j = -1; j <= 1; j ++ )
			{
				buff[index] += edge[index+(i+xoff)+(j+yoff)*width];
			}
		}
	}
}

__global__
void min_5x5_k( int * buff, int * warp, unsigned int width, unsigned int height )
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if( x == width-1 || y == height-1 || x == 0 || y == 0 )
	{
		return;
	}

	int index = (height-y-1)*width+x;

	int minimum = warp[index];
	for( int i = -2; i <= 2; i ++ )
	{
		for( int j = -2; j <= 2; j ++ )
		{
			minimum = min( minimum, warp[index+i+j*width] );
		}
	}
	buff[index] = minimum;
}

__global__
void edge_3x3_k( int * buff, int * warp, unsigned int width, unsigned int height )
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if( x == width-1 || y == height-1 || x == 0 || y == 0 )
	{
		return;
	}

	int index = (height-y-1)*width+x;

	int count = 0;
	for( int i = -1; i <= 1; i ++ )
	{
		for( int j = -1; j <= 1; j ++ )
		{
			count += warp[index+i+j*width] < 3000;
		}
	}
	buff[index] = ( count < 7 && warp[index] < 3000 ) ? 1 : 0;
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
		
	int warpindex = getIndex( wpx+1, wpy+1, width, rWidth+1, rHeight+1 );

	if( warpindex >= 0 && target == player )
		atomicMin( &(warp[warpindex]), (int)depth );
	else
		atomicMin( &(warp[cindex]), (int)3000 );
}

extern "C" void process_depth( dim3 dimGrid, dim3 dimBlock, float4 * depthRGBA, unsigned short * depthRAW, unsigned int width, unsigned int height, float xyscale, int target )
{
	cudaMemcpy( dRAW, depthRAW,  width * height * sizeof( unsigned short ), cudaMemcpyHostToDevice ); cutilCheckMsg("RAW Depth Transfer");
	clear_k<<< dimGrid, dimBlock >>> ( dWARP, dHUFF, dEDGE, hmax, width, height );
	process_depth_k <<< dimGrid, dimBlock >>> ( dWARP, dRAW, width, height, make_float2( xyscale ), target ); cutilCheckMsg("Depth Process");

	min_5x5_k <<< dimGrid, dimBlock >>> ( dBUFF, dWARP, width, height ); cutilCheckMsg("Min filt");
	max_5x5_k <<< dimGrid, dimBlock >>> ( dWARP, dBUFF, width, height ); cutilCheckMsg("Max filt");
	edge_3x3_k <<< dimGrid, dimBlock >>> ( dEDGE, dWARP, width, height ); cutilCheckMsg("Edge filt");
	hough_k <<< dimGrid, dimBlock >>> ( dHUFF, dEDGE, width, height ); cutilCheckMsg("Hough filt");
	find_head_k <<< dimGrid, dimBlock >>> ( hmax, dWARP, dHUFF, width, height ); cutilCheckMsg("Big Max");
	make_pretty_k <<< dimGrid, dimBlock >>> ( dRGBA, dWARP, dEDGE, dHUFF, hmax, width, height ); cutilCheckMsg("Convert to Image");
	
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
	cudaMalloc( &dBUFF, width * height * sizeof( int ) );
	cutilCheckMsg("CUDA Malloc");
	cudaMalloc( &dHUFF, width * height * sizeof( int ) );
	cutilCheckMsg("CUDA Malloc");
	cudaMalloc( &dEDGE, width * height * sizeof( int ) );
	cutilCheckMsg("CUDA Malloc");
	cudaMalloc( &hmax, sizeof( int ) );
	cutilCheckMsg("CUDA Malloc");
}