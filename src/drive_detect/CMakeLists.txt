# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
#cmake_minimum_required(VERSION 3.1)
cmake_minimum_required(VERSION 2.8.3)

##set basic configurations
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
#set(VIBRANTE FALSE)

set(SDK_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SDK_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
##project
project(drive_detect)

##system packages
find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)
#find_package(OpenCV REQUIRED)
#find_package(GLEW REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  rospy
  std_msgs
)

##basic configuration
include(ArchConfiguration)
include(CommonConfiguration)
include(SamplesConfiguration)
include(Samples3rdparty)
include(SamplesInstallConfiguration)
#set(SDK_BINARY_DIR ${CMAKE_BINARY_DIR})

##driveworks SDK
find_package(Driveworks REQUIRED)
include_directories(${Driveworks_INCLUDE_DIR})

set(LIBRARIES ${Driveworks_LIBRARIES})

#list(APPEND Driveworks_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})
list(APPEND LIBRARIES ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})
list(APPEND LIBRARIES ${CMAKE_THREAD_LIBS_INIT} dl rt)
list(APPEND LIBRARIES GLEW::glew_s)
list(APPEND LIBRARIES glfw)
list(APPEND LIBRARIES
        ${X11_LIBRARIES} ${X11_Xrandr_LIB}
        ${X11_Xcursor_LIB} ${X11_Xxf86vm_LIB}
        ${X11_Xinerama_LIB} ${X11_Xi_LIB}
        ${CMAKE_THREAD_LIBS_INIT}
)


#-------------------------------------------------------------------------------
# Samples
#-------------------------------------------------------------------------------
if(CMAKE_CROSSCOMPILING)
  set(DRIVEWORKS_DATAPATH "../data")
else()
  set(DRIVEWORKS_DATAPATH "${CMAKE_CURRENT_LIST_DIR}/../data")
endif()
include_directories(${SDK_BINARY_DIR}/configured/samples)
configure_file(src/common/DataPath.hpp.in
               ${SDK_BINARY_DIR}/configured/samples/common/DataPath.hpp)

 set(SAMPLES common;dnn)
 foreach(SAMPLE ${SAMPLES})
    add_subdirectory(src/${SAMPLE})
endforeach()

if(WIN32)
  # Handle runtime library dependence
  get_filename_component(DriveworksLibPath ${Driveworks_LIBRARY} DIRECTORY)
  set(LIBRARY_DIR ${DriveworksLibPath}/../bin)

  file(GLOB dll_files "${LIBRARY_DIR}/*.dll")
  foreach(file ${dll_files})
        file(COPY "${file}" DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug")
        file(COPY "${file}" DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release")
        install(FILES ${LIBRARY_DIR}/${file}
                DESTINATION "${CMAKE_INSTALL_PREFIX}/${SDK_SAMPLE_DESTINATION}"
        )
  endforeach()
endif()
#############
# SET THESE VARIABLES TO YOUR DRIVEWORKS LOCATIONS
#set(Driveworks_INCLUDE_DIR /usr/local/driveworks/include/)
#set(Driveworks_SAMPLE_DIR /usr/local/driveworks/samples/src/)



#catkin_package(
#  LIBRARIES ${PROJECT_NAME}
#)


#link_directories(
#  ${Driveworks_LIBRARY})
#MESSAGE("Driveworks include directory ${Driveworks_INCLUDE_DIR}")


#include_directories(
#  ${Driveworks_INCLUDE_DIR}
#  ${CUDA_INCLUDE_DIRS}
#  ${CMAKE_CURRENT_SOURCE_DIR}/src
#  ${GLEW_INCLUDE_DIR}
#  ${catkin_INCLUDE_DIRS}
#  ${Driveworks_SAMPLE_DIR}
#)


# TODO: add a FindNvmedia.cmake file for this? Why does it not exist?
include_directories(
#/usr/share/visionworks/sources/3rdparty/glfw3/include/
/usr/local/driveworks/samples/src/
/usr/local/driveworks/samples/src/dnn/
#/usr/local/driveworks/include/
)


set(SOURCES
 src/main.cpp
 src/common/SampleFramework.cpp
 src/common/ProgramArguments.cpp
 src/common/WindowGLFW.cpp
 src/common/ConsoleColor.cpp
 src/common/Checks.hpp
# src/common/Window.hpp
# src/common/Grid.hpp
# src/common/Grid.cpp
# src/common/Shader.cpp
# src/common/Shader.hpp
# src/common/GPUProgram.cpp
# src/common/GPUProgram.hpp
# src/common/WindowGLFW.hpp
# src/common/MathUtils.hpp
# src/common/MathUtils.cpp
# src/common/MouseView3D.cpp
# src/common/MouseView3D.hpp
# src/common/Mat4.hpp
# src/common/ConsoleColor.hpp
# src/common/StatsCounter.hpp
# src/common/CudaTimer.hpp
# src/common/ProfilerCUDA.hpp
# src/common/ProfilerCUDA.cpp

 src/dnn/dnn_common/DNNInference.cpp
 src/dnn/dnn_common/SensorIOCuda.cpp
)

##これ重要
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES img_read
#  CATKIN_DEPENDS cv_bridge image_transport roscpp rospy std_msgs
#  DEPENDS system_lib
)


#-------------------------------------------------------------------------------
# Final target
#-------------------------------------------------------------------------------
set(NODE_NAME ${PROJECT_NAME}_node)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(${NODE_NAME} ${SOURCES})
target_link_libraries(${NODE_NAME}
  ${catkin_LIBRARIES}
  ${LIBRARIES}
#  nvmedia  
#  ${GLEW_LIBRARY}
#  driveworks
#  GLESv2
#  EGL
#  drm
#  glfw3
#  ${CUDA_LIBRARY}
  ${OpenCV_LIBRARIES} 
)

