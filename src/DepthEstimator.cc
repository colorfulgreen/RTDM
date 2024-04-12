#include <mutex>
#include <thread>

#include <torch/torch.h>
#include <torch/script.h> // One-stop header.
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "DepthEstimator.h"
#include "DepthFusion.h"
#include "KeyFrame.h" 

namespace irtdm {

DepthEstimator::DepthEstimator(System* pSys, Atlas* pAtlas, const char* mvsModulePath, DepthDrawer* pDepthDrawer) :
  mpSystem(pSys), mpAtlas(pAtlas), mpDepthDrawer(pDepthDrawer), mbFinishRequested(false) {

  c10::InferenceMode guard;
  torch::autograd::AutoGradMode guard2(false);

  mModule = torch::jit::load(mvsModulePath);
}

DepthEstimator::~DepthEstimator() {
}

void DepthEstimator::Run() {

  while(1) {
    SetAcceptKeyFrames(false);

    if(mpAtlas->isImuInitialized() && CheckNewKeyFrames()) {
      int bOK = CallMVSNet();
      if(bOK) {}
    }

    SetAcceptKeyFrames(true);
    if(CheckFinish())
      break;

    usleep(100);
  }
  SetFinish();
}


bool DepthEstimator::CallMVSNet() {

  c10::InferenceMode guard;
  torch::autograd::AutoGradMode guard2(false);

  auto options = torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided)
    .device(torch::kCPU)
    .requires_grad(false);

  if (mHeight == -1 || mWidth == -1) {
    cv::Mat mImGray = mvNewKeyFrames[0]->mImGray; 
    mHeight = mImGray.rows;
    mWidth = mImGray.cols;
  } 

  int batch = 1;
  int ref_i = 3;
  torch::Tensor ref_img = torch::empty({batch,3,mHeight,mWidth}, options);
  torch::Tensor src_imgs = torch::empty({batch,mnViews-1,3,mHeight,mWidth}, options);
	auto ref_in = torch::empty({batch,3,3}, options);
	auto src_in = torch::empty({batch,mnViews-1,3,3}, options);
	auto ref_ex = torch::empty({batch,4,4}, options);
	auto src_ex = torch::empty({batch,mnViews-1,4,4}, options);
	auto depth_range = torch::empty({batch,2}, options);

	auto ref_in_a = ref_in.accessor<float, 3>(); 
	auto src_in_a = src_in.accessor<float, 4>(); 
	auto ref_ex_a = ref_ex.accessor<float, 3>(); 
	auto src_ex_a = src_ex.accessor<float, 4>(); 
	auto depth_range_a = depth_range.accessor<float, 2>();

  int size = mvNewKeyFrames.size();
  if (size < mnViews) {
    std::cout << "[WARN] mvNewKeyFrames.size() =" << mvNewKeyFrames.size() << std::endl;
    return false;
  }

  std::vector<cv::Mat> imRGBs;
  imRGBs.resize(mnViews);

  std::vector<double> src_timestamps;  

  KeyFrame* pKFRef;
  for (int vid=0; vid<mnViews; vid++) {
    KeyFrame* pKF = mvNewKeyFrames[vid];
    if ((pKF->mImGray).empty()) {
      std::cout << "[WARN] pKF->mImGray is empty!!!" << std::endl;
      continue;
    }

    cvtColor(pKF->mImGray, imRGBs[vid], cv::COLOR_GRAY2RGB);
    Eigen::Matrix4f pose = pKF->GetPose().matrix();

    cvtColor(imRGBs[vid], imRGBs[vid], cv::COLOR_RGB2BGR);
    (imRGBs[vid]).convertTo(imRGBs[vid], CV_32FC3, 1.0);


    if (vid == ref_i) { // reference image
      pKFRef = pKF;

      ref_img = torch::from_blob(imRGBs[vid].data, {imRGBs[vid].rows, imRGBs[vid].cols ,3}, options).clone();
      ref_img = ref_img.permute({2,0,1});
      ref_img = ref_img.unsqueeze(0).div(255.);

      for (int i=0; i<3; i++)
        for (int j=0; j<3; j++)
          ref_in_a[0][i][j] = (pKF->GetK())(i,j); 

      for (int i=0; i<4; i++)
        for (int j=0; j<4; j++)
          ref_ex_a[0][i][j] = pose(i,j);
    }

    else { // source images

      src_timestamps.push_back(pKF->mTimeStamp);

      int src_i = vid < ref_i ? vid : vid - 1;
      torch::Tensor src_img = torch::from_blob(imRGBs[vid].data, {imRGBs[vid].rows, imRGBs[vid].cols ,3}, options).clone();        
      src_img = src_img.permute({2,0,1}).div(255.);
      src_imgs[0][src_i] = src_img; 

      for (int i=0; i<3; i++)
        for (int j=0; j<3; j++)
          src_in_a[0][src_i][i][j] = (pKF->GetK())(i,j); 

      for (int i=0; i<4; i++)
        for (int j=0; j<4; j++)
          src_ex_a[0][src_i][i][j] = pose(i,j);
    }
  }

  // TODO
  depth_range_a[0][0] = 0.2;
  depth_range_a[0][1] = pKFRef->ComputeSceneMedianDepth(2) * 3;

  mInputs.clear();
  mInputs.emplace_back(ref_img.to(torch::kCUDA));
  mInputs.emplace_back(src_imgs.to(torch::kCUDA));
  mInputs.emplace_back(ref_in.to(torch::kCUDA));
  mInputs.emplace_back(src_in.to(torch::kCUDA));
  mInputs.emplace_back(ref_ex.to(torch::kCUDA));
  mInputs.emplace_back(src_ex.to(torch::kCUDA));
  mInputs.emplace_back(depth_range.to(torch::kCUDA));

  unique_lock<mutex> lock(mMutexMVSNet);
  auto pred = mModule.forward(mInputs).toTuple();
  auto depth_tensor = pred->elements()[0].toTensor().to(torch::kCPU);
  auto conf_tensor = pred->elements()[1].toTensor().to(torch::kCPU);

  mDepth = cv::Mat(mHeight, mWidth, CV_32F, depth_tensor.data_ptr());
  mDepth.copyTo(pKFRef->mDepth);

  mpDepthDrawer->Update(this);
  mpDepthFusion->InsertKeyFrame(pKFRef);

  EraseFirstKeyFrame();

  return true;
}


cv::Mat DepthEstimator::GetResult() {
  return mDepth;
}

void DepthEstimator::SetAcceptKeyFrames(bool flag)
{
  unique_lock<mutex> lock(mMutexAccept);
  mbAcceptKeyFrames=flag;
}

bool DepthEstimator::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void DepthEstimator::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mvNewKeyFrames.push_back(pKF);
}

bool DepthEstimator::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return (mvNewKeyFrames.size() >= mnViews);
}

void DepthEstimator::EraseFirstKeyFrame()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mvNewKeyFrames.erase(mvNewKeyFrames.begin());
}

void DepthEstimator::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool DepthEstimator::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void DepthEstimator::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool DepthEstimator::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

bool DepthEstimator::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop) {
        mbStopped = true;
        cout << "Depth Estimator STOP" << endl;
        return true;
    }
    return false;
}

bool DepthEstimator::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

void DepthEstimator::SetDepthFusion(DepthFusion* pDepthFusion)
{
    mpDepthFusion=pDepthFusion;
}

}
