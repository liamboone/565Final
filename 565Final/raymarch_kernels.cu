#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cutil_math.h>
#include <thrust\random.h>
#include <cmath>
#include "cutil_inline.h"

#define DEG_TO_RAD 0.017453292519943 
#define EPS        0.001
#define EPS1       0.01
#define PI         3.1415926535897932
#define HALFPI     1.5707963267948966
#define QUARTPI    0.7853981633974483
#define ROOTTHREE  0.57735027
#define HUGEVAL    1e20
#define MAX_ITER   32

__device__
float dBox( float3 p, float3 b )
{
	float3 d = make_float3( abs(p.x) - b.x, abs(p.y) - b.y, abs(p.z) - b.z );
	float3 m = make_float3( max(d.x,0.0), max(d.y,0.0), max(d.z,0.0) );
	return min( max( d.x, max( d.y, d.z ) ), 0.0 ) + length( m );
}

__device__
float d2Box( float3 p, float3 b )
{
	float box1 = dBox( p + make_float3( 0.5, 0.0, 0.0 ), b );
	float box2 = dBox( p - make_float3( 0.5, 0.0, 0.0 ) , b );
	return min(box1, box2);
}

__device__
float dFloor( float3 p )
{
	return dBox( p, make_float3( HUGEVAL, 0.2, HUGEVAL ) );
}

__device__
float dSphere( float3 p, float r )
{
	return length( p ) - r;
}

__device__
float dTorus( float3 p, float2 t )
{
  float2 q = make_float2(length(make_float2(p.x, p.y))-t.x,p.y);
  return length(q)-t.y;
}

__device__
float dCone( float3 p, float2 c )
{
    // c must be normalized
    float q = length(make_float2(p.x, p.y));
    return dot(c,make_float2(q,p.z));
}

__device__
float map( float3 p, int & mat )
{
	float dist = min(dBox( p, make_float3( 1, 1, 0.25 ) ), dSphere( p, 0.5 ) );
	
	mat = 1;
	
	return dist;
}

__device__
float rayMarch( float3 p, float3 view, int & mat )
{
	float dist;
	float totaldist = 0.0;
	
	for( int it = 0; it < MAX_ITER; it ++ )
	{
		dist = map( p, mat ) * 0.8;
		
		totaldist += dist;
		
		if( abs( dist ) < EPS )
		{
			break;
		}
		p += view * dist;
	}
	if( abs( dist ) > 1e-2 ) totaldist = -1.0;
	return totaldist;
}

__device__
float3 gradientNormal(float3 p) {
	int m;
	return normalize(make_float3(
		map(p + make_float3(EPS, 0, 0), m) - map(p - make_float3(EPS, 0, 0), m),
		map(p + make_float3(0, EPS, 0), m) - map(p - make_float3(0, EPS, 0), m),
		map(p + make_float3(0, 0, EPS), m) - map(p - make_float3(0, 0, EPS), m)));
}

__device__
float getAmbientOcclusion( float3 p, float3 dir )
{
	int m;
	float sample0 = map( p + 0.1 * dir, m ) / 0.1;
	float sample1 = map( p + 0.2 * dir, m ) / 0.2;
	float sample2 = map( p + 0.3 * dir, m ) / 0.3;
	float sample3 = map( p + 0.4 * dir, m ) / 0.4;
	return ( sample0*0.05+sample1*0.1+sample2*0.25+sample3*0.6 );
}

//Yanked from nop's code
// source: the.savage@hotmail.co.uk
#define SS_K  15.0
__device__
float getShadow (float3 pos, float3 toLight, float lightDist) {
  float shadow = 1.0;

  float t = EPS1;
  float dt;

  for(int i=0; i<MAX_ITER; ++i)
  {
    int m;
    dt = map(pos+(toLight*t), m) * 0.8;
    
    if(dt < EPS)    // stop if intersect object
      return 0.0;

    shadow = min(shadow, SS_K*(dt/t));
    
    t += dt;
    
    if(t > lightDist)   // stop if reach light
      break;
  }
  
  return clamp(shadow, 0.0, 1.0);
}
#undef SS_K

__device__
float3 rayCast( float3 pos, float3 lpos, float3 view, int & mat, float3 & newpos )
{
	float dist = rayMarch( pos, view, mat );
	float3 color = make_float3( 1.0 );
	
	if( mat == 1 )
	{
		color = make_float3( 1.0, 0.5, 0.5 );	
	}
	else if( mat == 2 )
	{
		color = make_float3( 0.5, 1.0, 0.5 );	
	}
	else if( mat == 3 )
	{
		color = make_float3( 0.5, 0.5, 1.0 );	
	}
	
	if( dist < 0.0 ) color = make_float3( 0.0 );
	else
	{ 
		newpos = pos + view*dist;
		float3 ldir = normalize( lpos - newpos );
		float ldist = length( lpos - newpos );
		float3 norm = gradientNormal( newpos );
		float diffuse = max( 0.0, dot( norm, ldir ) );
		float specular = pow( max( 0.0, dot( reflect( view, norm ), ldir ) ), 150.0 );
		float shadow = 1.0;//getShadow( newpos + 0.01*ldir, ldir, length( lpos - newpos ) );
		float AO = 1.0;//getAmbientOcclusion( newpos, norm );
		float fog = 1.0;//exp( -0.05*dist );
		color = color*(AO*(diffuse*shadow + 0.1)*fog);
		color += make_float3(5.0*specular*fog);
	}
	return color;
}

__global__
void render_k( float4 * out, float3 campos, unsigned int width, unsigned int height )
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if( x >= width || y >= height )
	{
		return;
	}

	float3 color;

	float3 lpos = { 0.0, 4.0, 7.0 };
	
	campos = campos - make_float3( 0, -50, 0 );

	float3 look = -campos;
	
	float2 position = ( make_float2(x,y) - make_float2(width,height) / 2.0 ) / height * sin( 45.0 * DEG_TO_RAD / 2.0 );

	look = normalize( look );

	float3 right = cross( look, make_float3( 0.0, 1.0, 0.0 ) );
	float3 up = cross( right, look );
	float3 view = normalize( look + position.x*right + position.y*up );
	
	float3 pos = campos + view;
	float3 newpos;
	int mat;
	
	color = rayCast( pos, lpos, view, mat, newpos );

	out[x+y*width] = make_float4( color, 1.0 );
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
