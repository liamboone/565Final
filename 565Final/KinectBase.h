#ifndef _LIAM_KINECTBASE_
#define _LIAM_KINECTBASE_

#include <Windows.h>
#include "NuiApi.h"

class KinectBase
{ 
public: 
	KinectBase();
	~KinectBase();
	HRESULT ConnectKinect();
	bool NextFrame();
	float * getDepthAsImage();
	unsigned short * getDepth();

	static const NUI_IMAGE_RESOLUTION cDepthResolution = NUI_IMAGE_RESOLUTION_320x240;

private:
	INuiSensor * m_nuiSensor;

	HANDLE m_depthStreamHandle;
	HANDLE m_nextDepthFrameEvent;
	
    long m_depthWidth;
    long m_depthHeight;

	unsigned short * m_depthD16;

	float * m_depthRGBA;
};


#endif