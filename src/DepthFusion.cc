#include <cuda_runtime.h>
#include <chrono>
#include <mutex>

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>

#include "Converter.h"
#include "DepthFusion.h"
#include "utils/rgbd_sensor.h"
#include "tsdfvh/tsdf_volume.h"

namespace irtdm {

float3 ToFloat3(float in[3]) {
  return float3{.x = in[0], .y = in[1], .z = in[2]};
}

DepthFusion::DepthFusion(const struct DepthFusionOptions &options, MeshDrawer *pMeshDrawer): mpMeshDrawer(pMeshDrawer) {
  sensor_ = std::make_unique<refusion::RgbdSensor>();
  sensor_->cx = options.cx;
  sensor_->cy = options.cy;
  sensor_->fx = options.fx;
  sensor_->fy = options.fy;
  sensor_->rows = options.height;
  sensor_->cols = options.width;
  sensor_->depth_factor = 5000;

  refusion::tsdfvh::TsdfVolumeOptions tsdf_options{
      .voxel_size = options.voxel_size,
      .num_buckets = options.num_buckets,
      .bucket_size = options.bucket_size,
      .num_blocks = options.num_blocks,
      .block_size = options.block_size,
      .max_sdf_weight = options.max_sdf_weight,
      .truncation_distance = options.truncation_distance,
      .max_sensor_depth = options.max_sensor_depth,
      .min_sensor_depth = options.min_sensor_depth,
      .num_render_streams = options.num_render_streams,
      .height = options.height,
      .width = options.width,
  };

  cudaMallocManaged(&volume_, sizeof(refusion::tsdfvh::TsdfVolume));
  volume_->Init(tsdf_options);

  dr_mesh_vert = (float *) malloc(sizeof(float) * dr_mesh_num_max * 3);
  dr_mesh_cols = (float *) malloc(sizeof(float) * dr_mesh_num_max * 3);

  float mesh_lower_corner[3] = {-10, -10, -10};
  float mesh_upper_corner[3] = {10, 10, 10};
}


DepthFusion::~DepthFusion() {
  cudaDeviceSynchronize();
  volume_->Free();
  cudaDeviceSynchronize();
  cudaFree(volume_);
}


void DepthFusion::Run() {

  mnUnprocessedId = 0;
  while(1) {
    if(CheckNewKeyFrames()) {
      KeyFrame* pKF = mvNewKeyFrames[mnUnprocessedId];
      KeyFrame* pPrevKF = pKF->mPrevKF;
      cv::Mat filteredDepth;
      
      if(pPrevKF != NULL && !pPrevKF->mDepth.empty() ) {
        Eigen::Matrix3f K = pKF->GetK();
        Eigen::Matrix4f poseRef = pKF->GetPose().matrix(), poseSrc = pPrevKF->GetPose().matrix();
        cv::Mat refDepth = pKF->mDepth, srcDepth = pPrevKF->mDepth;
        filteredDepth = CheckGeometricConsistency(refDepth, srcDepth, K, poseRef, poseSrc);
      } else {
        mnUnprocessedId++;
        continue;
      } 

      Eigen::Matrix<float, 4, 4, Eigen::RowMajor> Pw2cRowMajor = pKF->GetPose().inverse().matrix();
      cv::Mat imBGR;
      cvtColor(pKF->mImGray, imBGR, cv::COLOR_GRAY2BGR);

      refusion::float4x4 pose_cuda(Pw2cRowMajor.data());
      volume_->IntegrateScanAsync(*sensor_, imBGR.data, (float*) filteredDepth.data, pose_cuda);
      volume_->ExtractMeshAsync(ToFloat3(mesh_lower_corner), ToFloat3(mesh_upper_corner));
      volume_->GetMeshSync(dr_mesh_num_max, &dr_mesh_num, dr_mesh_vert, dr_mesh_cols);

      mpMeshDrawer->Update(dr_mesh_num, dr_mesh_vert, dr_mesh_cols);

      mnUnprocessedId++;

    } else {
      usleep(100);
    }
  }
}


void DepthFusion::Synchronize() {
  cudaDeviceSynchronize();
}


void DepthFusion::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mvNewKeyFrames.push_back(pKF);
}


bool DepthFusion::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return mvNewKeyFrames.size() > (mnUnprocessedId + 1);
}


};
