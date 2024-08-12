// // Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
// 引用自https://github.com/colmap/colmap/tree/main/src/colmap/feature

// 这段代码包含了两个主要部分：FeatureMatcher类和FeatureMatcherCache类
#pragma once // 防止头文件被多次包含，类似于传统的 include guard (#ifndef / #define)。

#include "colmap/feature/types.h"
#include "colmap/geometry/gps.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/database.h"
#include "colmap/scene/image.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/util/cache.h"
#include "colmap/util/types.h"

// 引用了标准库中的memory（用于智能指针）、mutex（用于多线程锁）、unordered_map（用于哈希映射）。
#include <memory>
#include <mutex>
#include <unordered_map>

namespace colmap {

class FeatureMatcher {
 // FeatureMatcher是一个虚基类，定义了特征匹配器的接口。
 public:
  // virtual ~FeatureMatcher() = default;：定义了虚析构函数，确保派生类正确析构。
  virtual ~FeatureMatcher() = default;

  // If the same matcher is used for matching multiple pairs of feature sets,
  // then the caller may pass a nullptr to one of the keypoint/descriptor
  // arguments to inform the implementation that the keypoints/descriptors are
  // identical to the previous call. This allows the implementation to skip e.g.
  // uploading data to GPU memory or pre-computing search data structures for
  // one of the descriptors.

  virtual void Match( // Match是一个纯虚函数，用于匹配两个特征描述符集合，结果存储在matches中。
      const std::shared_ptr<const FeatureDescriptors>& descriptors1,
      const std::shared_ptr<const FeatureDescriptors>& descriptors2,
      FeatureMatches* matches) = 0;

  virtual void MatchGuided( // MatchGuided也是一个纯虚函数，用于在指导条件（如最大误差）下匹配特征，结果存储在two_view_geometry中。
      double max_error,
      const std::shared_ptr<const FeatureKeypoints>& keypoints1,
      const std::shared_ptr<const FeatureKeypoints>& keypoints2,
      const std::shared_ptr<const FeatureDescriptors>& descriptors1,
      const std::shared_ptr<const FeatureDescriptors>& descriptors2,
      TwoViewGeometry* two_view_geometry) = 0;
};

// FeatureMatcherCache类用于缓存特征匹配数据，减少数据库访问。
// 构造函数FeatureMatcherCache接受缓存大小cache_size、数据库指针database以及可选的do_setup参数。
class FeatureMatcherCache {
 public:
  FeatureMatcherCache(size_t cache_size,
                      std::shared_ptr<Database> database,
                      bool do_setup = false);

  void Setup(); // Setup函数用于初始化缓存内容。

  // 这些函数是访问缓存内容的接口，分别用于获取相机、图像、位姿先验、特征关键点、特征描述符、匹配对以及所有图像ID。
  const Camera& GetCamera(camera_t camera_id) const;
  const Image& GetImage(image_t image_id) const;
  const PosePrior& GetPosePrior(image_t image_id) const;
  std::shared_ptr<FeatureKeypoints> GetKeypoints(image_t image_id);
  std::shared_ptr<FeatureDescriptors> GetDescriptors(image_t image_id);
  FeatureMatches GetMatches(image_t image_id1, image_t image_id2);
  std::vector<image_t> GetImageIds() const;

  // 这些布尔函数检查缓存或数据库中是否存在对应的特征数据。
  bool ExistsPosePrior(image_t image_id) const;
  bool ExistsKeypoints(image_t image_id);
  bool ExistsDescriptors(image_t image_id);

  bool ExistsMatches(image_t image_id1, image_t image_id2);
  bool ExistsInlierMatches(image_t image_id1, image_t image_id2);

  // 这些函数用于将特征匹配和两视图几何数据写入数据库，或者从数据库中删除匹配数据。
  void WriteMatches(image_t image_id1,
                    image_t image_id2,
                    const FeatureMatches& matches);
  void WriteTwoViewGeometry(image_t image_id1,
                            image_t image_id2,
                            const TwoViewGeometry& two_view_geometry);

  void DeleteMatches(image_t image_id1, image_t image_id2);
  void DeleteInlierMatches(image_t image_id1, image_t image_id2);

 // 私有成员变量用于管理缓存和数据库访问。
 private:
  const size_t cache_size_; // 缓存大小
  const std::shared_ptr<Database> database_; // 数据库指针
  std::mutex database_mutex_; // 用于多线程锁定数据库操作
  // cameras_cache_、images_cache_、locations_priors_cache_：分别用于缓存相机、图像和位姿先验数据的哈希映射。
  std::unordered_map<camera_t, Camera> cameras_cache_;
  std::unordered_map<image_t, Image> images_cache_;
  std::unordered_map<image_t, PosePrior> locations_priors_cache_;
  // keypoints_cache_、descriptors_cache_：用于缓存特征关键点和描述符的LRU缓存。
  std::unique_ptr<LRUCache<image_t, std::shared_ptr<FeatureKeypoints>>>
      keypoints_cache_;
  std::unique_ptr<LRUCache<image_t, std::shared_ptr<FeatureDescriptors>>>
      descriptors_cache_;
  // keypoints_exists_cache_、descriptors_exists_cache_：用于缓存特征关键点和描述符是否存在的布尔值LRU缓存。
  std::unique_ptr<LRUCache<image_t, bool>> keypoints_exists_cache_;
  std::unique_ptr<LRUCache<image_t, bool>> descriptors_exists_cache_;
};

}  // namespace colmap
