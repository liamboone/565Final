#include <iostream>

#include "KinectBase.h"

#include <GL/glew.h>
#include <GL/freeglut.h>

// CUDA standard includes
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

// CUDA helper functions
#include <sdkHelper.h>
#include <rendercheck_gl.h>
#include <cudaHelper.h>
#include <cudaGLHelper.h>
#include "cutil_inline.h"

extern "C" void process_depth( dim3 dimGrid, dim3 dimBlock, float4 * depthRGBA, unsigned short * depthRAW, unsigned int width, unsigned int height );
extern "C" void cudaInit();

using namespace std;

KinectBase * theJumpoff;

void initGL( int * argc, char * argv[] );
void display( );

int main( int argc, char* argv[] )
{
	int devID;
    cudaDeviceProp deviceProps;
    
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    devID = findCudaGLDevice(argc, argv);

    // get number of SMs on this GPU
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    cout << "CUDA device [" << deviceProps.name << "] has " << 
		deviceProps.multiProcessorCount << " Multi-Processors" << endl;


	initGL(&argc, argv);

	theJumpoff = new KinectBase();

	if( theJumpoff->ConnectKinect() == E_FAIL )
		cout << "NO BUENO" << endl;
	else
		cout << "CONNECTION SUCCESSFUL" << endl;

	cudaInit();

	glutMainLoop( );

	delete theJumpoff;

	return 0;
}

void initGL( int * argc, char * argv[] )
{
	glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GL_DOUBLE);
    glutInitWindowSize(320, 240);
    glutCreateWindow("565Final");
    
	glutDisplayFunc(display);

    glewInit();
    glViewport(0, 0, 320, 240);
	
	glClearColor(0.0, 0.0, 0.0, 1.0);

	glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho( -1, 1, -1, 1, -1, 1 );

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glDisable(GL_DEPTH_TEST);
}


void display()
{
	theJumpoff->NextFrame();
	dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(320 / dimBlock.x, 240 / dimBlock.y, 1);
	float4 * depthRGBA = new float4[320*240];

	process_depth( dimGrid, dimBlock, depthRGBA, theJumpoff->getDepth(), 320, 240 );

	glDrawPixels( 320, 240, GL_RGBA, GL_FLOAT, depthRGBA );
    
	delete[] depthRGBA;
	glutSwapBuffers();
	glutPostRedisplay();
}