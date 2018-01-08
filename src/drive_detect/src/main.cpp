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
#define _CRT_SECURE_NO_WARNINGS

#include <memory>
#include <thread>
#include <unordered_map>

// Sample
#include <common/DataPath.hpp>
#include <common/ProgramArguments.hpp>
#include <common/SampleFramework.hpp>

// CORE
#include <dw/core/Context.h>
#include <dw/core/Logger.h>

// Renderer
#include <dw/renderer/Renderer.h>

// SAL
#include <dw/sensors/Sensors.h>
#include <dw/sensors/camera/Camera.h>

// IMAGE
#include <dw/image/FormatConverter.h>
#include <dw/image/ImageStreamer.h>

// DNN
#include <dnn_common/DNNInference.hpp>

// Input/Output
#include <dnn_common/ISensorIO.hpp>
#include <dnn_common/SensorIOCuda.hpp>

#ifdef VIBRANTE
#include <dnn_common/SensorIONvmedia.hpp>
#endif

#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sstream>
#include <string>

#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
//------------------------------------------------------------------------------
// Variables
//------------------------------------------------------------------------------

dwContextHandle_t gSdk                   = DW_NULL_HANDLE;
dwSALHandle_t gSal                       = DW_NULL_HANDLE;
dwImageStreamerHandle_t gCuda2gl         = DW_NULL_HANDLE;
dwImageStreamerHandle_t gNvMedia2Cuda    = DW_NULL_HANDLE;
dwImageFormatConverterHandle_t gYuv2rgba = DW_NULL_HANDLE;
dwRendererHandle_t gRenderer             = DW_NULL_HANDLE;
dwRenderBufferHandle_t gLineBuffer       = DW_NULL_HANDLE;
dwSensorHandle_t gCameraSensor           = DW_NULL_HANDLE;

dwRect gScreenRectangle;

dwImageCUDA *gRgbaImage;
dwImageCUDA *gYuvImage;
dwImageCUDA Image;

std::unique_ptr<ISensorIO> gSensorIO;

uint32_t gCameraWidth  = 0U;
uint32_t gCameraHeight = 0U;

std::unique_ptr<DNNInference> gDnnInference;
std::vector<dwBox2D> gDnnBoxList;
//------------------------------------------------------------------------------

//#######################################################################################
void setupRenderer(dwRendererHandle_t &renderer, dwContextHandle_t dwSdk)
{
    dwRenderer_initialize(&renderer, dwSdk);

    float32_t rasterTransform[9];
    rasterTransform[0] = 1.0f;
    rasterTransform[3] = 0.0f;
    rasterTransform[6] = 0.0f;
    rasterTransform[1] = 0.0f;
    rasterTransform[4] = 1.0f;
    rasterTransform[7] = 0.0f;
    rasterTransform[2] = 0.0f;
    rasterTransform[5] = 0.0f;
    rasterTransform[8] = 1.0f;

    dwRenderer_set2DTransform(rasterTransform, renderer);
    float32_t boxColor[4] = {0.0f,1.0f,0.0f,1.0f};
    dwRenderer_setColor(boxColor, renderer);
    dwRenderer_setLineWidth(2.0f, renderer);
    dwRenderer_setRect(gScreenRectangle, renderer);
}

//#######################################################################################
void setupLineBuffer(dwRenderBufferHandle_t &lineBuffer, unsigned int maxLines, dwContextHandle_t dwSdk)
{
    dwRenderBufferVertexLayout layout;
    layout.posFormat   = DW_RENDER_FORMAT_R32G32_FLOAT;
    layout.posSemantic = DW_RENDER_SEMANTIC_POS_XY;
    layout.colFormat   = DW_RENDER_FORMAT_NULL;
    layout.colSemantic = DW_RENDER_SEMANTIC_COL_NULL;
    layout.texFormat   = DW_RENDER_FORMAT_NULL;
    layout.texSemantic = DW_RENDER_SEMANTIC_TEX_NULL;
    dwRenderBuffer_initialize(&lineBuffer, layout, DW_RENDER_PRIM_LINELIST, maxLines, dwSdk);
    dwRenderBuffer_set2DCoordNormalizationFactors((float32_t)gCameraWidth,
                                                  (float32_t)gCameraHeight, lineBuffer);
}

//#######################################################################################
void createVideoReplay(dwSensorHandle_t &salSensor,
                       uint32_t &cameraWidth,
                       uint32_t &cameraHeight,
                       uint32_t &cameraSiblings,
                       float32_t &cameraFrameRate,
                       dwImageType &imageType,
                       dwSALHandle_t sal,
                       const std::string &videoFName)
{
#ifdef VIBRANTE
    auto yuv2rgb = gArguments.get("yuv2rgb");
    std::string arguments = "video=" + videoFName + ",yuv2rgb=" + yuv2rgb;
#else
    std::string arguments = "video=" + videoFName;
#endif

    dwSensorParams params;
    params.parameters = arguments.c_str();
    params.protocol   = "camera.virtual";
    dwSAL_createSensor(&salSensor, params, sal);

    dwImageProperties cameraImageProperties{};
    dwSensorCamera_getImageProperties(&cameraImageProperties,
                                      DW_CAMERA_PROCESSED_IMAGE,
                                      salSensor);
    dwCameraProperties cameraProperties{};
    dwSensorCamera_getSensorProperties(&cameraProperties, salSensor);

    cameraWidth = cameraImageProperties.width;
    cameraHeight = cameraImageProperties.height;
    imageType = cameraImageProperties.type;
    cameraFrameRate = cameraProperties.framerate;
    cameraSiblings = cameraProperties.siblings;

}

//#######################################################################################
void initDriveworks()
{
    // create a Logger to log to console
    // we keep the ownership of the logger at the application level
    dwLogger_initialize(getConsoleLoggerCallback(true));
    dwLogger_setLogLevel(DW_LOG_VERBOSE);

    // instantiate Driveworks SDK context
    dwContextParameters sdkParams = {};

#ifdef VIBRANTE
    sdkParams.eglDisplay = gWindow->getEGLDisplay();
#endif

    dwInitialize(&gSdk, DW_VERSION, &sdkParams);
}

//#######################################################################################
void initRenderer()
{
    // init renderer
    gScreenRectangle.height = gWindow->height();
    gScreenRectangle.width = gWindow->width();
    gScreenRectangle.x = 0;
    gScreenRectangle.y = 0;

    unsigned int maxLines = 20000;
    setupRenderer(gRenderer, gSdk);
    setupLineBuffer(gLineBuffer, maxLines, gSdk);
}

//#######################################################################################
void initCameras()
{
    // create sensor abstraction layer
    dwSAL_initialize(&gSal, gSdk);

    // create GMSL Camera interface
    uint32_t cameraSiblings   = 0U;
    float32_t cameraFramerate = 0.0f;
    dwImageType imageType;

    createVideoReplay(gCameraSensor, gCameraWidth, gCameraHeight, cameraSiblings,
                      cameraFramerate, imageType, gSal,
                      gArguments.has("video") ? gArguments.get("video") : "");

    std::cout << "Camera image with " << gCameraWidth << "x" << gCameraHeight << " at "
              << cameraFramerate << " FPS" << std::endl;

#ifdef VIBRANTE
    gSensorIO.reset(new SensorIONvmedia(gSdk, 0, gCameraSensor, gCameraWidth, gCameraHeight));
#else
    gSensorIO.reset(new SensorIOCuda(gSdk, 0, gCameraSensor, gCameraWidth, gCameraHeight));
#endif

    gRun = gRun && dwSensor_start(gCameraSensor) == DW_SUCCESS;
}

//#######################################################################################
void initDNN()
{
    gDnnInference.reset(new DNNInference(gSdk));

    if (!gArguments.has("tensorRT_model")) {
        std::string caffePrototxt = gArguments.get("caffe_prototxt");
        std::string caffeModel    = gArguments.get("caffe_model");
        std::cout << "Initializing Caffe Network: " << caffePrototxt.c_str() << ", "
                  << caffeModel.c_str() << std::endl;
        gDnnInference->buildFromCaffe(caffePrototxt.c_str(), caffeModel.c_str());
    } else {
        std::string tensorRTModel = gArguments.get("tensorRT_model");
        std::cout << "Initializing TensorRT Network: " << tensorRTModel.c_str() << std::endl;
        gDnnInference->buildFromTensorRT(tensorRTModel.c_str());
    }
}

//#######################################################################################
void init()
{
    initDriveworks();

    initCameras();

    initRenderer();

    initDNN();
}

//#######################################################################################
void release()
{
    dwSensor_stop(gCameraSensor);
    dwSAL_releaseSensor(&gCameraSensor);

    dwRenderBuffer_release(&gLineBuffer);
    dwRenderer_release(&gRenderer);

    // release used objects in correct order
    gSensorIO.reset();
    dwSAL_release(&gSal);
    dwRelease(&gSdk);
    dwLogger_release();
    gDnnInference.reset();
}

//#######################################################################################
void runDetector()
{
    // Run inference if the model is valid
    if (gDnnInference->isLoaded()) {
        gDnnBoxList.clear();
        gDnnInference->inferSingleFrame(&gDnnBoxList, &Image, true);
/*
        uint32_t height,width; 
        height = Image.prop.height;
        width = Image.prop.width;
	printf("heigit=%d,width=%d\n",height,width);
        int layout;       
        layout = Image.layout;
	printf("layout=%d\n",layout);
*/        

//        std::vector<dwRect>::const_iterator i = gDnnBoxList.begin ();
//        printf ("height=%d,width=%d,x=%d,y=%d,\n", i->height,i->width,i->x,i->y);

        for (std::vector<dwRect>::const_iterator i = gDnnBoxList.begin ();i != gDnnBoxList.end (); i++)
        {
//                (void) printf ("height=%d,width=%d,x=%d,y=%d,\n", i->height,i->width,i->x,i->y);
        }
        drawBoxes(gDnnBoxList, NULL, static_cast<float32_t>(gCameraWidth),
                  static_cast<float32_t>(gCameraHeight), gLineBuffer, gRenderer);
    }
}

//#######################################################################################
int main(int argc, char **argv)
{
    const ProgramArguments arguments = ProgramArguments(
        {
#ifdef WINDOWS
            ProgramArguments::Option_t("caffe_prototxt",
                                       (DataPath::get() +
                                        std::string{"/samples/detector/predict.prototxt"})
                                           .c_str()),
            ProgramArguments::Option_t("caffe_model",
                                       (DataPath::get() +
                                        std::string{"/samples/detector/weights.caffemodel"})
                                           .c_str()),
#else
            ProgramArguments::Option_t("tensorRT_model",
                                        (DataPath::get() +
                                         std::string{"/samples/detector/tensorRT_model.bin"})
                                            .c_str()),
#endif
            ProgramArguments::Option_t("video",
                                       (DataPath::get() +
                                        std::string{"/samples/sfm/triangulation/video_0.h264"})
                                           .c_str()),
#ifdef VIBRANTE
            ProgramArguments::Option_t("yuv2rgb", "cuda"),
#endif
        });

    // init framework
    int nv_argc=3;
    const char *nv_argv[] = {(const char*)"./sample_object_detector",(const char*)"--tensorRT_model=/home/mine/work/driveworks_test/samples/../data/samples/detector/tensorRT_model.bin",(const char*)"--video=/home/mine/work/driveworks_test/samples/../data/samples/sfm/triangulation/video_0.h264"};
/*
       for (int i = 0; i < nv_argc; i++)
              printf("nv_argv[%d] = %s\n", i, nv_argv[i]);
*/
    initSampleApp(nv_argc,nv_argv, &arguments, NULL, 1280, 800);
/*
    printf("argc= %d\n", argc);
       int i;
       for (i = 0; i < argc; i++)
              printf("argv[%d] = %s\n", i, argv[i]);
*/

    // init driveworks
    init();

  ros::init(argc, argv, "test");
  printf("argc= %d\n", argc);
       int i;
       for (i = 0; i < argc; i++)
              printf("argv[%d] = %s\n", i, argv[i]);
  ros::NodeHandle n;
/*
  std::string a;

  n.getParam("a", a);
  std::cout << "parameter : " << a << std::endl;
*/
  image_transport::ImageTransport it(n);
  image_transport::Publisher pub = it.advertise("image",1);

    ros::Rate loop_rate(1);

   cv::Mat rgb_image = cv::imread("/home/mine/work/test/ros_nv/src/drive_detect/src/image.png",1);
   //  cv::imshow("Image", image);
   //  cv::waitKey(30);
   int width = rgb_image.size().width;
   int height = rgb_image.size().height;

    // RGBA image to display

    dwImageProperties Properties;
    Properties.height = height;
    Properties.meta.analogGain= 0;
    Properties.meta.conversionGain= 0;
    Properties.meta.digitalGain= 0;
    Properties.meta.embeddedDataSize.x= 0;
    Properties.meta.embeddedDataSize.y= 0;
    Properties.meta.exposureTime= 0;
    Properties.meta.flags= 0;
    Properties.meta.wbGain[4]= 0;
    Properties.planeCount= 1;
    Properties.pxlFormat= DW_IMAGE_RGBA;
    Properties.pxlType= DW_TYPE_UINT8;
    Properties.type= DW_IMAGE_CUDA;
    Properties.width = width;


    dwStatus result = DW_FAILURE;
    result = dwImageCUDA_create(&Image, &Properties, DW_IMAGE_CUDA_PITCH);
    if (result != DW_SUCCESS) {
        std::cerr << "Cannot create RGBA CUDA image" << std::endl;
    }


uint8_t PixelArray[4 * width * height];
for (int y = 0; y < height; y++) {
 for (int x = 0; x < width; x++) {
    cv::Vec3b p = rgb_image.at<cv::Vec3b>(y, x);
    PixelArray[(width*4)*(y)+x*4+0] = p[2];
    PixelArray[(width*4)*(y)+x*4+1] = p[1];
    PixelArray[(width*4)*(y)+x*4+2] = p[0];
    PixelArray[(width*4)*(y)+x*4+3] = 0;
 }
}
// copy values
cudaMemcpy(Image.dptr[0], &PixelArray,
		sizeof(uint8_t) * 4 * 1280 * 800, cudaMemcpyHostToDevice);

//
dwImageGL frameGL;
dwImageProperties GLProperties=Properties;
GLProperties.type= DW_IMAGE_GL;
dwImageGL_create(&frameGL,&GLProperties,GL_TEXTURE_2D);
dwImageGL_setupTexture(&frameGL,PixelArray,0,0);
dwRenderer_renderTexture(frameGL.tex, frameGL.target, gRenderer);
//

while (gRun && !gWindow->shouldClose()) {
//        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        std::this_thread::yield();
        runDetector();
        gWindow->swapBuffers();
        CHECK_GL_ERROR();
}
    release();

  while (ros::ok())
  {
//    std_msgs::String msg;
//    std::stringstream ss;
//    ss << "hello world " << count;
//    msg.data = ss.str();
//    ROS_INFO("%s", msg.data.c_str());
//    chatter_pub.publish(msg);
//    pub.publish(msg);
    ros::spinOnce();


//    loop_rate.sleep();
//    ++count;
  }
  return 0;
}
