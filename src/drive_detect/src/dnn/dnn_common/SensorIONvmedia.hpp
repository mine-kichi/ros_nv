/////////////////////////////////////////////////////////////////////////////////////////
// This code contains NVIDIA Confidential Information and is disclosed
// under the Mutual Non-Disclosure Agreement.
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//
// NVIDIA Corporation assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA Corporation. No third party distribution is allowed unless
// expressly authorized by NVIDIA.  Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA Corporation products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA Corporation.
//
// Copyright (c) 2015-2016 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef SAMPLE_CAMERA_TRACKER_NVMEDIA_HPP__
#define SAMPLE_CAMERA_TRACKER_NVMEDIA_HPP__

#include "ISensorIO.hpp"

#include <dw/sensors/camera/Camera.h>
#include <dw/sensors/camera/Camera_vibrante.h>
#include <dw/image/FormatConverter.h>
#include <dw/image/ImageStreamer.h>

#include <vector>

class SensorIONvmedia : public ISensorIO
{
  public:
    SensorIONvmedia(dwContextHandle_t context, cudaStream_t stream, dwSensorHandle_t cameraSensor,
                int cameraWidth, int cameraHeight);
    ~SensorIONvmedia();

    dwStatus getFrame() override final;

    dwImageCUDA *getCudaYuv() override final;
    void releaseCudaYuv() override final;

    dwImageCUDA *getCudaRgba() override final;
    void releaseCudaRgba() override final;

    dwImageGL *getGlRgbaFrame() override final;
    void releaseGLRgbaFrame() override final;

    void releaseFrame() override final;

  protected:
    //////////////////////////
    // Configuration members
    NvMediaDevice *m_nvmedia;
    cudaStream_t m_cudaStream;
    dwSensorHandle_t m_sensor;
    dwImageFormatConverterHandle_t m_yuv2rgbaCUDA;
    dwImageFormatConverterHandle_t m_yuv2rgbaNVM;
    dwImageStreamerHandle_t m_nvm2gl;
    dwImageStreamerHandle_t m_nvm2cudaYuv;

    static const int POOL_SIZE = 2;
    std::vector<dwImageNvMedia *> m_rgbaImagePool;

    ////////////////////////
    // Current frame
    dwCameraFrameHandle_t m_frameHandle;
    dwImageNvMedia *m_frameNvmYuv;
    dwImageNvMedia *m_frameNvmRgba;
    dwImageCUDA *m_frameCUDAyuv;
    dwImageCUDA *m_frameCUDArgba;
    dwImageGL *m_frameGlRgba;
};

#endif
