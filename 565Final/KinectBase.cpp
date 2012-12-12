#include "KinectBase.h"
#include <cmath>

// Safe release for interfaces
template<class Interface>
inline void SafeRelease( Interface *& pInterfaceToRelease )
{
    if ( pInterfaceToRelease != NULL )
    {
        pInterfaceToRelease->Release();
        pInterfaceToRelease = NULL;
    }
}

KinectBase::KinectBase() :
    m_nextDepthFrameEvent(INVALID_HANDLE_VALUE),
    m_depthStreamHandle(INVALID_HANDLE_VALUE),
    m_nuiSensor(NULL)
{
	unsigned long width = 0;
    unsigned long height = 0;

    NuiImageResolutionToSize(cDepthResolution, width, height);
    m_depthWidth  = static_cast<long>(width);
    m_depthHeight = static_cast<long>(height);
	
    m_depthD16 = new unsigned short[m_depthWidth*m_depthHeight];
	m_depthRGBA = new float[m_depthWidth*m_depthHeight*4];
}

KinectBase::~KinectBase()
{
    if (m_nuiSensor)
    {
        m_nuiSensor->NuiShutdown();
    }

    if (m_nextDepthFrameEvent != INVALID_HANDLE_VALUE)
    {
        CloseHandle(m_nextDepthFrameEvent);
    }

    // done with pixel data
    delete[] m_depthD16;
	delete[] m_depthRGBA;

    SafeRelease(m_nuiSensor);
}

HRESULT KinectBase::ConnectKinect()
{
    INuiSensor * pNuiSensor;
    HRESULT hr;

    int iSensorCount = 0;
    hr = NuiGetSensorCount(&iSensorCount);
    if (FAILED(hr))
    {
        return hr;
    }

    // Look at each Kinect sensor
    for (int i = 0; i < iSensorCount; ++i)
    {
        // Create the sensor so we can check status, if we can't create it, move on to the next
        hr = NuiCreateSensorByIndex(i, &pNuiSensor);
        if (FAILED(hr))
        {
            continue;
        }

        // Get the status of the sensor, and if connected, then we can initialize it
        hr = pNuiSensor->NuiStatus();
        if (S_OK == hr)
        {
            m_nuiSensor = pNuiSensor;
            break;
        }

        // This sensor wasn't OK, so release it since we're not using it
        pNuiSensor->Release();
    }

    if (NULL != m_nuiSensor)
    {
        // Initialize the Kinect and specify that we'll be using depth
        hr = m_nuiSensor->NuiInitialize(NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX); 
        if (SUCCEEDED(hr))
        {
            // Create an event that will be signaled when depth data is available
            m_nextDepthFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

            // Open a depth image stream to receive depth frames
            hr = m_nuiSensor->NuiImageStreamOpen(
                NUI_IMAGE_TYPE_DEPTH_AND_PLAYER_INDEX,
                cDepthResolution,
                0,
                2,
                m_nextDepthFrameEvent,
                &m_depthStreamHandle);

			m_nuiSensor->NuiImageStreamSetImageFrameFlags( 
				m_depthStreamHandle, 
				NUI_IMAGE_STREAM_FLAG_ENABLE_NEAR_MODE );
        }
    }

    if (NULL == m_nuiSensor || FAILED(hr))
    {
        return E_FAIL;
    }

    m_nuiSensor->NuiSkeletonTrackingDisable();

	m_nuiSensor->NuiCameraElevationSetAngle( 15 );

	float DegreesToRadians = 3.14159265359f / 180.0f;
	m_xyScale = tanf(NUI_CAMERA_DEPTH_NOMINAL_HORIZONTAL_FOV * DegreesToRadians * 0.5f) / (m_depthWidth * 0.5f);

    return hr;
}

bool KinectBase::NextFrame()
{
	HRESULT hr = S_OK;
    NUI_IMAGE_FRAME imageFrame;

	hr = m_nuiSensor->NuiImageStreamGetNextFrame(m_depthStreamHandle, 0, &imageFrame);
    if (FAILED(hr))
    {
        return false;
    }
	INuiFrameTexture * pTexture = imageFrame.pFrameTexture;
    NUI_LOCKED_RECT LockedRect;

    // Lock the frame data so the Kinect knows not to modify it while we're reading it
    pTexture->LockRect(0, &LockedRect, NULL, 0);

    // Make sure we've received valid data
    if (LockedRect.Pitch != 0)
    {
        memcpy(m_depthD16, LockedRect.pBits, LockedRect.size);
    }

    // We're done with the texture so unlock it
    pTexture->UnlockRect(0);

    // Release the frame
    m_nuiSensor->NuiImageStreamReleaseFrame(m_depthStreamHandle, &imageFrame);
	return true;
}

float * KinectBase::getDepthAsImage()
{
	for( int i = 0; i < m_depthWidth*m_depthHeight; i ++ )
	{
		int rgbaIdx = i*4;
		unsigned short playerIdx = NuiDepthPixelToPlayerIndex( m_depthD16[i] );
		unsigned short depth = NuiDepthPixelToDepth( m_depthD16[i] );
		if( playerIdx > 0 )
		{
			m_depthRGBA[rgbaIdx+0] = 1.0;
			m_depthRGBA[rgbaIdx+1] = 1.0;
			m_depthRGBA[rgbaIdx+2] = 1.0;
			m_depthRGBA[rgbaIdx+3] = 1.0;
		}
		else
		{
			float fdepth = depth == 0 ? 0.0 : ( 1.0 - ( depth - 400.0 ) / 3000.0 );
			m_depthRGBA[rgbaIdx+0] = fdepth;
			m_depthRGBA[rgbaIdx+1] = fdepth;
			m_depthRGBA[rgbaIdx+2] = fdepth;
			m_depthRGBA[rgbaIdx+3] = 1.0;
		}
	} 
	return m_depthRGBA;
}

unsigned short * KinectBase::getDepth()
{
	return m_depthD16;
}