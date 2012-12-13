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

extern "C" void process_depth( dim3 dimGrid, dim3 dimBlock, float4 * depthRGBA, unsigned short * depthRAW, unsigned int width, unsigned int height, int3 * head, int target );
extern "C" void reset( dim3 dimGrid, dim3 dimBlock, unsigned int width, unsigned int height );
extern "C" void cudaInit( unsigned int width, unsigned int height );
extern "C" void render( dim3 dimGrid, dim3 dimBlock, float4 * out, float3 campos, unsigned int width, unsigned int height );

using namespace std;

KinectBase * theJumpoff;

void initGL( int * argc, char * argv[] );
void display( );
void display2( );
void reshape (int w, int h);
void keyboard( unsigned char key, int, int );

int window1, window2;
int dwidth;
int dheight;
int player = 2;

float3 cameye;

int main( int argc, char* argv[] )
{
	theJumpoff = new KinectBase();

	if( theJumpoff->ConnectKinect() == E_FAIL )
		cout << "NOPE" << endl;
	else
		cout << "CONNECTION SUCCESSFUL" << endl;

	dwidth = theJumpoff->getWidth();
	dheight = theJumpoff->getHeight();

	cout << "Depth Stream: [" << dwidth << ", " << dheight << "]" << endl;

	int devID;
    cudaDeviceProp deviceProps;
    
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    devID = findCudaGLDevice(argc, argv);

    // get number of SMs on this GPU
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    cout << "CUDA device [" << deviceProps.name << "] has " << 
		deviceProps.multiProcessorCount << " Multi-Processors" << endl;


	initGL(&argc, argv);
	cudaInit( dwidth, dheight );

	glutMainLoop( );

	delete theJumpoff;

	return 0;
}

void initGL( int * argc, char * argv[] )
{
	glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GL_DOUBLE);
    glutInitWindowSize(dwidth, dheight);
    window1 = glutCreateWindow("565Final");

	glutDisplayFunc(display2);
	glutReshapeFunc(reshape);

    window2 = glutCreateWindow("Viewer");
    
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);

    glewInit();
    glViewport(0, 0, dwidth, dheight);
	
	glClearColor(0.0, 0.0, 0.0, 1.0);

	glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho( -1, 1, -1, 1, -1, 1 );

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glDisable(GL_DEPTH_TEST);
}

void display2(void)
{
	dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(dwidth / dimBlock.x, dheight / dimBlock.y, 1);
	float4 * tmp = new float4[dwidth*dheight];

	render( dimGrid, dimBlock, tmp, cameye, dwidth, dheight );

	glDrawPixels( dwidth, dheight, GL_RGBA, GL_FLOAT, tmp );
    
	delete[] tmp;
	glutSwapBuffers();
	glutPostRedisplay();
}

void reshape (int w, int h)
{
	glViewport(0,0,(GLsizei)w,(GLsizei)h);
	glutPostRedisplay();
}

void keyboard( unsigned char key, int, int )
{
	switch (key) 
	{
	   case '1':
		   cout << "Tracking player 1" << endl;
		   player = 1;
		   break;
	   case '2':
		   cout << "Tracking player 2" << endl;
		   player = 2;
		   break;
	   case '3':
		   cout << "Tracking player 3" << endl;
		   player = 3;
		   break;
	   case '4':
		   cout << "Tracking player 4" << endl;
		   player = 4;
		   break;
	   case '5':
		   cout << "Tracking player 5" << endl;
		   player = 5;
		   break;
	   case '6':
		   cout << "Tracking player 6" << endl;
		   player = 6;
		   break;
	   case '7':
		   cout << "Tracking player 7" << endl;
		   player = 7;
		   break;
	   case '[':
		   theJumpoff->setAngle( theJumpoff->getAngle() - 1.0 );
		   break;
	   case ']':
		   theJumpoff->setAngle( theJumpoff->getAngle() + 1.0 );
		   break;
	   case '{':
		   theJumpoff->setAngle( theJumpoff->getAngle() - 5.0 );
		   break;
	   case '}':
		   theJumpoff->setAngle( theJumpoff->getAngle() + 5.0 );
		   break;
	   default:
		   player = 0;
		   break;
	}
}

void display()
{
	theJumpoff->NextFrame();
	int3 head;
	dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(dwidth / dimBlock.x, dheight / dimBlock.y, 1);
	float4 * depthRGBA = new float4[dwidth*dheight];

	reset( dimGrid, dimBlock, dwidth, dheight );

	for( int i = 1; i <= 7; i++ )
	{
		process_depth( dimGrid, dimBlock, depthRGBA, theJumpoff->getDepth(), dwidth, dheight, &head, i );

		if( head.x != -1 && head.y != -1 && head.z != -1 )
		{
			cout << "Player " << i << " Head @[" << head.x << ", " << head.y << ", " << head.z << "]" << endl; 
		}
	}

	glDrawPixels( dwidth, dheight, GL_RGBA, GL_FLOAT, depthRGBA );
    
	delete[] depthRGBA;
	glutSwapBuffers();
	glutPostRedisplay();
}