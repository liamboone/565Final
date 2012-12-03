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
	glDrawPixels( 320, 240, GL_RGBA, GL_FLOAT, theJumpoff->getDepthAsImage() );
    glutSwapBuffers();
	glutPostRedisplay();
}