// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
// 引用自https://github.com/colmap/colmap/blob/main/src/colmap/feature/matcher.cc
// 这段代码实现了一个特征匹配缓存(FeatureMatcherCache)的类，用于优化图像特征数据的存取效率。
// 这个类通过缓存相机、图像、关键点、描述符、位姿先验等信息，减少多次从数据库读取相同数据的开销。
#include "colmap/feature/matcher.h"

namespace colmap {

// 这是FeatureMatcherCache类的构造函数，用于初始化一个FeatureMatcherCache对象。
FeatureMatcherCache::FeatureMatcherCache(const size_t cache_size,
                                         std::shared_ptr<Database> database,
                                         const bool do_setup)
    : cache_size_(cache_size), // cache_size_（成员变量）保存传入的缓存大小。
    // database_（成员变量）存储传入的数据库指针，
    // THROW_CHECK_NOTNULL(database)用于确保database不是空指针，如果是空指针，将抛出异常。
      database_(std::move(THROW_CHECK_NOTNULL(database))) {
  if (do_setup) {// 如果do_setup为true，则调用Setup()方法来初始化缓存。
    Setup();
  }
}

// 定义了Setup()方法的开始。这个方法初始化所有与缓存相关的数据结构。
void FeatureMatcherCache::Setup() {
  // 从数据库读取所有的相机数据并存储在cameras向量中。
  std::vector<Camera> cameras = database_->ReadAllCameras();
  // 使用reserve方法预留足够的空间，以避免向cameras_cache_插入时频繁分配内存。
  cameras_cache_.reserve(cameras.size());
  // 遍历所有相机数据，将它们的camera_id和相机对象一起存入cameras_cache_，
  // 这里使用了emplace方法，它高效地插入元素到缓存中，避免不必要的拷贝。
  for (Camera& camera : cameras) {
    cameras_cache_.emplace(camera.camera_id, std::move(camera));
  }

  std::vector<Image> images = database_->ReadAllImages();
  images_cache_.reserve(images.size());
  for (Image& image : images) {
    images_cache_.emplace(image.ImageId(), std::move(image));
  }
  
  // locations_priors_cache_用于存储图像的位姿先验信息。根据数据库中的位姿先验数量预留缓存空间。
  locations_priors_cache_.reserve(database_->NumPosePriors());
  // 遍历images_cache_中的每个图像，检查是否存在对应的位姿先验。
  // 如果存在，则从数据库中读取该位姿先验信息，并存入locations_priors_cache_中。
  for (const auto& id_and_image : images_cache_) {
    if (database_->ExistsPosePrior(id_and_image.first)) {
      locations_priors_cache_.emplace(
          id_and_image.first, database_->ReadPosePrior(id_and_image.first));
    }
  }

  // keypoints_cache_是一个LRU缓存，用于存储图像的关键点数据。
  keypoints_cache_ =
	  // 使用std::make_unique创建一个唯一指针（unique_ptr），指向一个LRU缓存对象，缓存大小为cache_size_。
      // 当缓存未命中时，回调函数会从数据库读取图像的关键点数据并将其放入缓存。
	  std::make_unique<LRUCache<image_t, std::shared_ptr<FeatureKeypoints>>>(
          cache_size_, [this](const image_t image_id) {
            return std::make_shared<FeatureKeypoints>(
                database_->ReadKeypoints(image_id));
          });

  // descriptors_cache_是另一个LRU缓存，用于存储图像的描述符数据。
  descriptors_cache_ =
      std::make_unique<LRUCache<image_t, std::shared_ptr<FeatureDescriptors>>>(
          cache_size_, [this](const image_t image_id) {
            return std::make_shared<FeatureDescriptors>(
                database_->ReadDescriptors(image_id));
          });

  // keypoints_exists_cache_用于缓存某个图像是否存在关键点的布尔值。
  keypoints_exists_cache_ = std::make_unique<LRUCache<image_t, bool>>(
      images_cache_.size(), [this](const image_t image_id) {
        return database_->ExistsKeypoints(image_id);
      });

  // descriptors_exists_cache_类似于keypoints_exists_cache_，用于缓存图像描述符是否存在。
  descriptors_exists_cache_ = std::make_unique<LRUCache<image_t, bool>>(
      images_cache_.size(), [this](const image_t image_id) {
        return database_->ExistsDescriptors(image_id);
      });
}

// GetCamera方法从cameras_cache_中返回指定camera_id的相机对象的引用。如果找不到对应的相机ID，会抛出异常。
const Camera& FeatureMatcherCache::GetCamera(const camera_t camera_id) const {
  return cameras_cache_.at(camera_id);
}

// GetImage方法从images_cache_中返回指定image_id的图像对象的引用。
const Image& FeatureMatcherCache::GetImage(const image_t image_id) const {
  return images_cache_.at(image_id);
}

// GetPosePrior方法从locations_priors_cache_中返回指定图像ID的位姿先验信息。
const PosePrior& FeatureMatcherCache::GetPosePrior(
    const image_t image_id) const {
  return locations_priors_cache_.at(image_id);
}

// GetKeypoints方法从keypoints_cache_中获取指定图像的关键点数据。如果数据不在缓存中，回调函数会从数据库中读取数据。
// std::lock_guard<std::mutex>用于线程安全，确保多个线程不会同时访问数据库。
std::shared_ptr<FeatureKeypoints> FeatureMatcherCache::GetKeypoints(
    const image_t image_id) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  return keypoints_cache_->Get(image_id);
}

// GetDescriptors方法类似于GetKeypoints，用于获取图像的描述符数据。
std::shared_ptr<FeatureDescriptors> FeatureMatcherCache::GetDescriptors(
    const image_t image_id) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  return descriptors_cache_->Get(image_id);
}

// GetMatches方法用于从数据库中读取两个图像之间的特征匹配数据。
FeatureMatches FeatureMatcherCache::GetMatches(const image_t image_id1,
                                               const image_t image_id2) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  return database_->ReadMatches(image_id1, image_id2);
}

// GetImageIds方法返回缓存中所有图像的ID列表。
// 它首先为image_ids向量预留足够的空间，然后将所有图像ID添加到向量中并返回。
std::vector<image_t> FeatureMatcherCache::GetImageIds() const {
  std::vector<image_t> image_ids;
  image_ids.reserve(images_cache_.size());
  for (const auto& image : images_cache_) {
    image_ids.push_back(image.first);
  }
  return image_ids;
}

// ExistsPosePrior方法检查指定图像的位姿先验是否存在于缓存中。
bool FeatureMatcherCache::ExistsPosePrior(const image_t image_id) const {
  return locations_priors_cache_.find(image_id) !=
         locations_priors_cache_.end();
}

// ExistsKeypoints方法检查指定图像的关键点数据是否存在。
bool FeatureMatcherCache::ExistsKeypoints(const image_t image_id) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  return keypoints_exists_cache_->Get(image_id);
}

// ExistsDescriptors方法检查指定图像的描述符数据是否存在。
bool FeatureMatcherCache::ExistsDescriptors(const image_t image_id) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  return descriptors_exists_cache_->Get(image_id);
}

// ExistsMatches方法检查数据库中是否存在image_id1和image_id2之间的特征匹配数据。
bool FeatureMatcherCache::ExistsMatches(const image_t image_id1,
                                        const image_t image_id2) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  return database_->ExistsMatches(image_id1, image_id2);
}

// ExistsInlierMatches方法检查数据库中是否存在image_id1和image_id2之间的内点匹配数据（即在几何验证后被认为是正确的匹配）。
bool FeatureMatcherCache::ExistsInlierMatches(const image_t image_id1,
                                              const image_t image_id2) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  return database_->ExistsInlierMatches(image_id1, image_id2);
}

// WriteMatches方法将image_id1和image_id2之间的特征匹配数据写入数据库。
void FeatureMatcherCache::WriteMatches(const image_t image_id1,
                                       const image_t image_id2,
                                       const FeatureMatches& matches) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  database_->WriteMatches(image_id1, image_id2, matches);
}

// WriteTwoViewGeometry方法将image_id1和image_id2之间的两视图几何数据写入数据库。
void FeatureMatcherCache::WriteTwoViewGeometry(
    const image_t image_id1,
    const image_t image_id2,
    const TwoViewGeometry& two_view_geometry) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  database_->WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
}

// DeleteMatches方法用于从数据库中删除image_id1和image_id2之间的特征匹配数据。
void FeatureMatcherCache::DeleteMatches(const image_t image_id1,
                                        const image_t image_id2) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  database_->DeleteMatches(image_id1, image_id2);
}

// DeleteInlierMatches方法用于从数据库中删除image_id1和image_id2之间的内点匹配数据。
void FeatureMatcherCache::DeleteInlierMatches(const image_t image_id1,
                                              const image_t image_id2) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  database_->DeleteInlierMatches(image_id1, image_id2);
}

}  // namespace colmap
