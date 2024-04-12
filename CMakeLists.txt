cmake_minimum_required(VERSION 2.8)
project(irtdm)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " $CMAKE_BUILD_TYPE)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -g3  -O0 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -w -g3 -O0 -fPIC")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

set(CMAKE_CXX_STANDARD 17)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 4.4.0 REQUIRED)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 4.4 not found.")
   endif()
MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

find_package(Eigen3 3.4.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Torch REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/include/CameraModels
${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
${PROJECT_SOURCE_DIR}/Thirdparty/g2o
${PROJECT_SOURCE_DIR}/Thirdparty/refusion/src
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${OpenCV_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
add_library(${PROJECT_NAME} SHARED 
    src/Atlas.cc
    src/CameraModels/Pinhole.cpp
    src/Converter.cc
    src/DepthDrawer.cc
    src/DepthEstimator.cc
    src/DepthFusion.cc
    src/Frame.cc
    src/FrameDrawer.cc
    src/G2oTypes.cc
    src/GeometricTools.cc
    src/ImuTypes.cc
    src/LocalMapping.cc
    src/LoopClosing.cc
    src/KeyFrame.cc
    src/KeyFrameDatabase.cc
    src/Map.cc
    src/MapDrawer.cc
    src/MapPoint.cc
    src/MeshDrawer.cc
    src/MLPnPsolver.cpp
    src/OptimizableTypes.cpp
    src/Optimizer.cc
    src/ORBextractor.cc
    src/ORBmatcher.cc
    src/Settings.cc
    src/System.cc
    src/Sim3Solver.cc
    src/Tracking.cc
    src/TwoViewReconstruction.cc
    src/System.cc
    src/Viewer.cc
    include/Atlas.h
    include/Converter.h
    include/DepthDrawer.h
    include/DepthEstimator.h
    include/DepthFusion.h
    include/Frame.h
    include/FrameDrawer.h
    include/G2oTypes.h
    include/GeometricTools.h
    include/ImuTypes.h
    include/KeyFrame.h
    include/KeyFrameDatabase.h
    include/LocalMapping.h
    include/LoopClosing.h
    include/Map.h
    include/MapDrawer.h
    include/MapPoint.h
    include/MeshDrawer.h
    include/MLPnPsolver.h
    include/OptimizableTypes.h
    include/Optimizer.h
    include/ORBextractor.h
    include/ORBmatcher.h
    include/SerializationUtils.h
    include/Settings.h
    include/Sim3Solver.h
    include/System.h
    include/TwoViewReconstruction.h
    include/Tracking.h
    include/Viewer.h
    include/CameraModels/GeometricCamera.h
    include/CameraModels/Pinhole.h
)

# add_subdirectory(Thirdparty/g2o)
add_subdirectory(Thirdparty/refusion)
add_subdirectory("ASMVSNet/net/layers/deform_conv")

target_link_libraries(${PROJECT_NAME}
    ${EIGEN3_LIBS}
    ${Pangolin_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
    ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o_core.so
    ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o_types_sim3.so
    ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o_types_slam3d.so
    refusion
    -lboost_serialization
    -lcrypto
)
target_link_libraries(${PROJECT_NAME} -Wl,--no-as-needed 
    deform_conv_cuda)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular-Inertial)
add_executable(mono_inertial_euroc
               Examples/Monocular-Inertial/mono_inertial_euroc.cc)
target_link_libraries(mono_inertial_euroc ${PROJECT_NAME})
