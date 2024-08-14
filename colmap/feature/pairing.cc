// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#include "colmap/feature/pairing.h"

#include "colmap/feature/utils.h"
#include "colmap/geometry/gps.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

#include <fstream>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace colmap {
namespace {

// ReadImagePairsText: 这是一个用于从文本文件读取图像配对信息的函数。
// 函数读取文件中的每一行，解析图像名称，并检查这些图像是否存在于给定的图像名称到图像ID的映射中。它会忽略无效的或重复的配对。
std::vector<std::pair<image_t, image_t>> ReadImagePairsText(
    const std::string& path, // path 是输入文件的路径，存储了图像对信息。
    // image_name_to_image_id 是一个映射表，将图像名称映射到对应的图像ID。
    const std::unordered_map<std::string, image_t>& image_name_to_image_id) {
  std::ifstream file(path);
  THROW_CHECK_FILE_OPEN(file, path);

  std::string line;
  // 返回值是一个包含图像ID配对的向量 std::vector<std::pair<image_t, image_t>>。
  std::vector<std::pair<image_t, image_t>> image_pairs;
  // 定义一个无序集合 image_pairs_set，用于存储已经读取的图像对ID，以便检查重复。
  std::unordered_set<image_pair_t> image_pairs_set;
  while (std::getline(file, line)) {
    StringTrim(&line); // 这行代码移除读取行两端的空白字符。

    if (line.empty() || line[0] == '#') {
      continue;
    }
    // 使用 line_stream 将读取的行转换为一个流，以便进一步处理。
    std::stringstream line_stream(line);

    std::string image_name1; 
    std::string image_name2;

    std::getline(line_stream, image_name1, ' '); // // 从流中提取第一个图像名称（以空格为分隔符）。
    StringTrim(&image_name1); // 移除提取出的图像名称两端的空白字符。
    std::getline(line_stream, image_name2, ' ');
    StringTrim(&image_name2);

    // 检查 image_name1 是否在 image_name_to_image_id 中存在。如果不存在，记录错误并跳过这一行。
    if (image_name_to_image_id.count(image_name1) == 0) {
      LOG(ERROR) << "Image " << image_name1 << " does not exist.";
      continue;
    }
    if (image_name_to_image_id.count(image_name2) == 0) {
      LOG(ERROR) << "Image " << image_name2 << " does not exist.";
      continue;
    }

    // 获取 image_name1 对应的图像ID。
    const image_t image_id1 = image_name_to_image_id.at(image_name1);
    const image_t image_id2 = image_name_to_image_id.at(image_name2);
    // 调用 ImagePairToPairId 函数，将两个图像ID组合成一个唯一的图像对ID。
    const image_pair_t image_pair =
        Database::ImagePairToPairId(image_id1, image_id2);
    // 尝试将图像对ID插入到 image_pairs_set 集合中。如果插入成功（即这个图像对之前未出现），image_pair_exists 为 true。
    const bool image_pair_exists = image_pairs_set.insert(image_pair).second;
    // 如果图像对是新配对，且不重复，则将图像ID对添加到 image_pairs 向量中。
    if (image_pair_exists) {
      image_pairs.emplace_back(image_id1, image_id2);
    }
  }
  return image_pairs;
}

}  // namespace

// 这段代码定义了一个名为 ExhaustiveMatchingOptions::Check() 的成员函数，
// 用于验证 ExhaustiveMatchingOptions 类中的某些配置选项是否合法。
bool ExhaustiveMatchingOptions::Check() const {
  CHECK_OPTION_GT(block_size, 1);
  return true;
}

// 这段代码定义了一个名为 VocabTreeMatchingOptions::Check() 的成员函数，
// 用于验证 VocabTreeMatchingOptions 类中的配置选项是否合法。
bool VocabTreeMatchingOptions::Check() const {
  CHECK_OPTION_GT(num_images, 0);
  CHECK_OPTION_GT(num_nearest_neighbors, 0);
  CHECK_OPTION_GT(num_checks, 0);
  return true;
}

bool SequentialMatchingOptions::Check() const {
  CHECK_OPTION_GT(overlap, 0);
  CHECK_OPTION_GT(loop_detection_period, 0);
  CHECK_OPTION_GT(loop_detection_num_images, 0);
  CHECK_OPTION_GT(loop_detection_num_nearest_neighbors, 0);
  CHECK_OPTION_GT(loop_detection_num_checks, 0);
  return true;
}

// 这段代码定义了一个名为 VocabTreeOptions 的成员函数，属于 SequentialMatchingOptions 类。
// 它的作用是从当前对象的成员变量中构造并返回一个 VocabTreeMatchingOptions 类型的对象，
// 该对象包含了用于词汇树匹配的所有必要参数。
VocabTreeMatchingOptions SequentialMatchingOptions::VocabTreeOptions() const {
  VocabTreeMatchingOptions options;
  options.num_images = loop_detection_num_images;
  options.num_nearest_neighbors = loop_detection_num_nearest_neighbors;
  options.num_checks = loop_detection_num_checks;
  options.num_images_after_verification =
      loop_detection_num_images_after_verification;
  options.max_num_features = loop_detection_max_num_features;
  options.vocab_tree_path = vocab_tree_path;
  return options;
}

bool SpatialMatchingOptions::Check() const {
  CHECK_OPTION_GT(max_num_neighbors, 0);
  CHECK_OPTION_GT(max_distance, 0.0);
  return true;
}

bool TransitiveMatchingOptions::Check() const {
  CHECK_OPTION_GT(batch_size, 0);
  CHECK_OPTION_GT(num_iterations, 0);
  return true;
}

bool ImagePairsMatchingOptions::Check() const {
  CHECK_OPTION_GT(block_size, 0);
  return true;
}

bool FeaturePairsMatchingOptions::Check() const { return true; } 

// 这段代码定义了 PairGenerator 类的一个成员函数 AllPairs，它的作用是生成所有图像对，
// 并将它们存储在一个 std::vector<std::pair<image_t, image_t>> 类型的容器中。
// 这个函数通过反复调用 Next() 函数来获取图像对，并将它们合并到最终的结果向量中。
std::vector<std::pair<image_t, image_t>> PairGenerator::AllPairs() {
  // 返回类型是 std::vector<std::pair<image_t, image_t>>，表示该函数返回一个包含图像对的向量。
  std::vector<std::pair<image_t, image_t>> image_pairs;
  while (!this->HasFinished()) {
    std::vector<std::pair<image_t, image_t>> image_pairs_block = this->Next();
    image_pairs.insert(image_pairs.end(),
                       std::make_move_iterator(image_pairs_block.begin()),
                       std::make_move_iterator(image_pairs_block.end()));
  }
  return image_pairs;
}

// 这段代码定义了 ExhaustivePairGenerator 类的构造函数。
// 它的主要功能是根据提供的选项和缓存数据初始化一个用于生成图像对的生成器。
ExhaustivePairGenerator::ExhaustivePairGenerator(
    const ExhaustiveMatchingOptions& options,
    const std::shared_ptr<FeatureMatcherCache>& cache)
    : options_(options),
      image_ids_(THROW_CHECK_NOTNULL(cache)->GetImageIds()),
      block_size_(static_cast<size_t>(options_.block_size)),
      num_blocks_(static_cast<size_t>(
          std::ceil(static_cast<double>(image_ids_.size()) / block_size_))) {
  THROW_CHECK(options.Check());
  LOG(INFO) << "Generating exhaustive image pairs...";
  const size_t num_pairs_per_block = block_size_ * (block_size_ - 1) / 2;
  image_pairs_.reserve(num_pairs_per_block);
}

// 这段代码定义了 ExhaustivePairGenerator 类的另一个构造函数。
// 这个构造函数通过调用之前定义的构造函数来初始化对象，并且它使用了一个 FeatureMatcherCache 对象。
ExhaustivePairGenerator::ExhaustivePairGenerator(
    const ExhaustiveMatchingOptions& options,
    const std::shared_ptr<Database>& database)
    : ExhaustivePairGenerator(
          options,
          std::make_shared<FeatureMatcherCache>(CacheSize(options),
                                                THROW_CHECK_NOTNULL(database),
                                                /*do_setup=*/true)) {}
// 定义了 ExhaustivePairGenerator 类中的 Reset 方法。
// 这个方法的作用是重置图像配对生成器的状态，使得生成器可以重新开始生成图像对。
void ExhaustivePairGenerator::Reset() {
  start_idx1_ = 0;
  start_idx2_ = 0;
}

// 这段代码定义了 ExhaustivePairGenerator 类中的 HasFinished 方法，用于判断图像对生成器是否已经完成了所有图像对的生成。
bool ExhaustivePairGenerator::HasFinished() const {
  return start_idx1_ >= image_ids_.size();
}

// 这段代码定义了 ExhaustivePairGenerator 类的 Next 方法，该方法生成并返回下一批图像对。
// ExhaustivePairGenerator 类用于在一组图像中生成所有可能的图像对，而 Next 方法是分批生成图像对的核心逻辑。
std::vector<std::pair<image_t, image_t>> ExhaustivePairGenerator::Next() {
  image_pairs_.clear();
  if (HasFinished()) {
    return image_pairs_;
  }

  const size_t end_idx1 =
      std::min(image_ids_.size(), start_idx1_ + block_size_) - 1;
  const size_t end_idx2 =
      std::min(image_ids_.size(), start_idx2_ + block_size_) - 1;

  LOG(INFO) << StringPrintf("Matching block [%d/%d, %d/%d]",
                            start_idx1_ / block_size_ + 1,
                            num_blocks_,
                            start_idx2_ / block_size_ + 1,
                            num_blocks_);

  for (size_t idx1 = start_idx1_; idx1 <= end_idx1; ++idx1) {
    for (size_t idx2 = start_idx2_; idx2 <= end_idx2; ++idx2) {
      const size_t block_id1 = idx1 % block_size_;
      const size_t block_id2 = idx2 % block_size_;
      if ((idx1 > idx2 && block_id1 <= block_id2) ||
          (idx1 < idx2 && block_id1 < block_id2)) {  // Avoid duplicate pairs
        image_pairs_.emplace_back(image_ids_[idx1], image_ids_[idx2]);
      }
    }
  }
  start_idx2_ += block_size_;
  if (start_idx2_ >= image_ids_.size()) {
    start_idx2_ = 0;
    start_idx1_ += block_size_;
  }
  return image_pairs_;
}

// 这段代码定义了 VocabTreePairGenerator 类的构造函数，用于通过词汇树生成图像配对。
// 构造函数根据给定的配置选项和图像缓存，初始化生成器，并准备将所有图像索引到词汇树中。
VocabTreePairGenerator::VocabTreePairGenerator(
    // 该构造函数接受三个参数：options（词汇树匹配选项）、cache（特征匹配缓存）、以及 query_image_ids（查询图像ID列表）。
    const VocabTreeMatchingOptions& options,
    std::shared_ptr<FeatureMatcherCache> cache,
    const std::vector<image_t>& query_image_ids)
    : options_(options),
      cache_(std::move(THROW_CHECK_NOTNULL(cache))),
      thread_pool(options_.num_threads),
      queue(options_.num_threads) {
  THROW_CHECK(options.Check());
  LOG(INFO) << "Generating image pairs with vocabulary tree...";

  // Read the pre-trained vocabulary tree from disk.
  visual_index_.Read(options_.vocab_tree_path);

  const std::vector<image_t> all_image_ids = cache_->GetImageIds();
  if (query_image_ids.size() > 0) {
    query_image_ids_ = query_image_ids;
  } else if (options_.match_list_path == "") {
    query_image_ids_ = cache_->GetImageIds();
  } else {
    // Map image names to image identifiers.
    std::unordered_map<std::string, image_t> image_name_to_image_id;
    image_name_to_image_id.reserve(all_image_ids.size());
    for (const auto image_id : all_image_ids) {
      const auto& image = cache_->GetImage(image_id);
      image_name_to_image_id.emplace(image.Name(), image_id);
    }

    // Read the match list path.
    std::ifstream file(options_.match_list_path);
    THROW_CHECK_FILE_OPEN(file, options_.match_list_path);
    std::string line;
    while (std::getline(file, line)) {
      StringTrim(&line);

      if (line.empty() || line[0] == '#') {
        continue;
      }

      if (image_name_to_image_id.count(line) == 0) {
        LOG(ERROR) << "Image " << line << " does not exist.";
      } else {
        query_image_ids_.push_back(image_name_to_image_id.at(line));
      }
    }
  }

  IndexImages(all_image_ids); // 将所有图像ID索引到词汇树中。

  query_options_.max_num_images = options_.num_images; // 设置查询选项中的最大图像数量。
  query_options_.num_neighbors = options_.num_nearest_neighbors; // 设置查询选项中的最近邻数量。
  query_options_.num_checks = options_.num_checks; // 设置查询选项中的检查次数。
  query_options_.num_images_after_verification =
      options_.num_images_after_verification; // 设置查询选项中在验证后保留的最大图像数量。
}

// 这个构造函数通过接收匹配选项、数据库指针和查询图像ID列表，来初始化一个 VocabTreePairGenerator 对象。
// 在初始化时，它调用另一个构造函数，并在此过程中创建了一个 FeatureMatcherCache 对象。
VocabTreePairGenerator::VocabTreePairGenerator(
    const VocabTreeMatchingOptions& options,
    const std::shared_ptr<Database>& database,
    const std::vector<image_t>& query_image_ids)
    : VocabTreePairGenerator(
          options,
          std::make_shared<FeatureMatcherCache>(CacheSize(options),
                                                THROW_CHECK_NOTNULL(database),
                                                /*do_setup=*/true),
          query_image_ids) {}

void VocabTreePairGenerator::Reset() {
  query_idx_ = 0;
  result_idx_ = 0;
}

bool VocabTreePairGenerator::HasFinished() const {
  return result_idx_ >= query_image_ids_.size();
}

std::vector<std::pair<image_t, image_t>> VocabTreePairGenerator::Next() {
  image_pairs_.clear();
  if (HasFinished()) {
    return image_pairs_;
  }
  if (query_idx_ == 0) {
    // Initially, make all retrieval threads busy and continue with the
    // matching.
    const size_t init_num_tasks =
        std::min(query_image_ids_.size(), 2 * thread_pool.NumThreads());
    for (; query_idx_ < init_num_tasks; ++query_idx_) {
      thread_pool.AddTask(
          &VocabTreePairGenerator::Query, this, query_image_ids_[query_idx_]);
    }
  }

  LOG(INFO) << StringPrintf(
      "Matching image [%d/%d]", result_idx_ + 1, query_image_ids_.size());

  // Push the next image to the retrieval queue.
  if (query_idx_ < query_image_ids_.size()) {
    thread_pool.AddTask(
        &VocabTreePairGenerator::Query, this, query_image_ids_[query_idx_++]);
  }

  // Pop the next results from the retrieval queue.
  auto retrieval = queue.Pop();
  THROW_CHECK(retrieval.IsValid());

  const auto& image_id = retrieval.Data().image_id;
  const auto& image_scores = retrieval.Data().image_scores;

  // Compose the image pairs from the scores.
  image_pairs_.reserve(image_scores.size());
  for (const auto image_score : image_scores) {
    image_pairs_.emplace_back(image_id, image_score.image_id);
  }
  ++result_idx_;
  return image_pairs_;
}

// IndexImages 函数的主要任务是将一组图像的特征信息添加到视觉索引中，以便于后续的图像检索操作。
// 它通过多个步骤，包括特征提取、特征过滤、索引添加和索引准备，来确保图像检索的效率和准确性。
void VocabTreePairGenerator::IndexImages(
    const std::vector<image_t>& image_ids) {
  retrieval::VisualIndex<>::IndexOptions index_options;
  // num_threads 和 num_checks 是从 options_ 对象中获取的配置，用于指定索引过程使用的线程数和检查次数
  index_options.num_threads = options_.num_threads;
  index_options.num_checks = options_.num_checks;

  // 遍历图像ID并索引每个图像
  for (size_t i = 0; i < image_ids.size(); ++i) {
    Timer timer;
    timer.Start();
    LOG(INFO) << StringPrintf(
        "Indexing image [%d/%d]", i + 1, image_ids.size());
    // 获取关键点和描述符
    auto keypoints = *cache_->GetKeypoints(image_ids[i]);
    auto descriptors = *cache_->GetDescriptors(image_ids[i]);
    if (options_.max_num_features > 0 &&
        descriptors.rows() > options_.max_num_features) {
      ExtractTopScaleFeatures(
          &keypoints, &descriptors, options_.max_num_features);
    }
    // 将图像的特征添加到视觉索引中
    visual_index_.Add(index_options, image_ids[i], keypoints, descriptors);
    LOG(INFO) << StringPrintf(" in %.3fs", timer.ElapsedSeconds());
  }

  // Compute the TF-IDF weights, etc.
  visual_index_.Prepare();
}

// Query 函数的主要任务是对指定的图像进行查询操作，并将查询结果存储到一个队列中。
// 它首先从缓存中获取图像的特征信息，然后根据配置对特征进行筛选，最后调用视觉索引进行查询，并将结果安全地推送到一个队列中，以便后续处理。
void VocabTreePairGenerator::Query(const image_t image_id) {
  auto keypoints = *cache_->GetKeypoints(image_id);
  auto descriptors = *cache_->GetDescriptors(image_id);
  if (options_.max_num_features > 0 &&
      descriptors.rows() > options_.max_num_features) {
    ExtractTopScaleFeatures(
        &keypoints, &descriptors, options_.max_num_features);
  }
  // 执行查询操作
  Retrieval retrieval;
  retrieval.image_id = image_id;
  visual_index_.Query(
      query_options_, keypoints, descriptors, &retrieval.image_scores);

  THROW_CHECK(queue.Push(std::move(retrieval)));
}

// 这个构造函数用于初始化一个顺序图像对生成器 (SequentialPairGenerator) 对象，
// 该对象可能用于图像序列的匹配或相似图像对的生成。
SequentialPairGenerator::SequentialPairGenerator(
    const SequentialMatchingOptions& options,
    std::shared_ptr<FeatureMatcherCache> cache)
    : options_(options), cache_(std::move(THROW_CHECK_NOTNULL(cache))) {
  THROW_CHECK(options.Check());
  LOG(INFO) << "Generating sequential image pairs...";
  image_ids_ = GetOrderedImageIds();
  image_pairs_.reserve(options_.overlap);

  if (options_.loop_detection) {
    std::vector<image_t> query_image_ids;
    for (size_t i = 0; i < image_ids_.size();
         i += options_.loop_detection_period) {
      query_image_ids.push_back(image_ids_[i]);
    }
    vocab_tree_pair_generator_ = std::make_unique<VocabTreePairGenerator>(
        options_.VocabTreeOptions(), cache_, query_image_ids);
  }
}

// 这段代码实现了 SequentialPairGenerator 类的一个重载构造函数，用于在给定选项和数据库的情况下，初始化一个 SequentialPairGenerator 对象。
SequentialPairGenerator::SequentialPairGenerator(
    const SequentialMatchingOptions& options,
    const std::shared_ptr<Database>& database)
    : SequentialPairGenerator(
          options,
          std::make_shared<FeatureMatcherCache>(CacheSize(options),
                                                THROW_CHECK_NOTNULL(database),
                                                /*do_setup=*/true)) {}

void SequentialPairGenerator::Reset() {
  image_idx_ = 0;
  if (vocab_tree_pair_generator_) {
    vocab_tree_pair_generator_->Reset();
  }
}

bool SequentialPairGenerator::HasFinished() const {
  return image_idx_ >= image_ids_.size() &&
         (vocab_tree_pair_generator_ ? vocab_tree_pair_generator_->HasFinished()
                                     : true);
}

std::vector<std::pair<image_t, image_t>> SequentialPairGenerator::Next() {
  image_pairs_.clear();
  if (image_idx_ >= image_ids_.size()) {
    if (vocab_tree_pair_generator_) {
      return vocab_tree_pair_generator_->Next();
    }
    return image_pairs_;
  }
  LOG(INFO) << StringPrintf(
      "Matching image [%d/%d]", image_idx_ + 1, image_ids_.size());

  const auto image_id1 = image_ids_.at(image_idx_);
  for (int i = 0; i < options_.overlap; ++i) {
    const size_t image_idx_2 = image_idx_ + i;
    if (image_idx_2 < image_ids_.size()) {
      image_pairs_.emplace_back(image_id1, image_ids_.at(image_idx_2));
      if (options_.quadratic_overlap) {
        const size_t image_idx_2_quadratic = image_idx_ + (1ull << i);
        if (image_idx_2_quadratic < image_ids_.size()) {
          image_pairs_.emplace_back(image_id1,
                                    image_ids_.at(image_idx_2_quadratic));
        }
      }
    } else {
      break;
    }
  }
  ++image_idx_;
  return image_pairs_;
}

std::vector<image_t> SequentialPairGenerator::GetOrderedImageIds() const {
  const std::vector<image_t> image_ids = cache_->GetImageIds();

  std::vector<Image> ordered_images;
  ordered_images.reserve(image_ids.size());
  for (const auto image_id : image_ids) {
    ordered_images.push_back(cache_->GetImage(image_id));
  }

  std::sort(ordered_images.begin(),
            ordered_images.end(),
            [](const Image& image1, const Image& image2) {
              return image1.Name() < image2.Name();
            });

  std::vector<image_t> ordered_image_ids;
  ordered_image_ids.reserve(image_ids.size());
  for (const auto& image : ordered_images) {
    ordered_image_ids.push_back(image.ImageId());
  }

  return ordered_image_ids;
}

// SpatialPairGenerator 构造函数的主要任务是基于图像的空间位置信息生成图像对。
// 它通过读取位置数据、构建索引和执行最近邻搜索来实现这一点。搜索结果存储在矩阵中，以便后续用于图像配对和匹配任务。
SpatialPairGenerator::SpatialPairGenerator(
    const SpatialMatchingOptions& options,
    const std::shared_ptr<FeatureMatcherCache>& cache)
    : options_(options), image_ids_(cache->GetImageIds()) {
  LOG(INFO) << "Generating spatial image pairs...";
  THROW_CHECK(options.Check());

  Timer timer;
  timer.Start();
  LOG(INFO) << "Indexing images...";

  Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> location_matrix =
      ReadLocationData(*cache);
  size_t num_locations = location_idxs_.size();

  LOG(INFO) << StringPrintf(" in %.3fs", timer.ElapsedSeconds());
  if (num_locations == 0) {
    LOG(INFO) << "=> No images with location data.";
    return;
  }

  timer.Restart();
  LOG(INFO) << "Building search index...";

  flann::Matrix<float> locations(
      location_matrix.data(), num_locations, location_matrix.cols());

  flann::LinearIndexParams index_params;
  flann::LinearIndex<flann::L2<float>> search_index(index_params);
  search_index.buildIndex(locations);

  LOG(INFO) << StringPrintf(" in %.3fs", timer.ElapsedSeconds());

  timer.Restart();
  LOG(INFO) << "Searching for nearest neighbors...";

  const int knn = std::min<int>(options_.max_num_neighbors, num_locations);
  image_pairs_.reserve(knn);

  index_matrix_.resize(num_locations, knn);
  flann::Matrix<size_t> indices(index_matrix_.data(), num_locations, knn);

  distance_matrix_.resize(num_locations, knn);
  flann::Matrix<float> distances(distance_matrix_.data(), num_locations, knn);

  flann::SearchParams search_params(flann::FLANN_CHECKS_AUTOTUNED);
  if (options_.num_threads == ThreadPool::kMaxNumThreads) {
    search_params.cores = std::thread::hardware_concurrency();
  } else {
    search_params.cores = options_.num_threads;
  }
  if (search_params.cores <= 0) {
    search_params.cores = 1;
  }

  search_index.knnSearch(locations, indices, distances, knn, search_params);

  LOG(INFO) << StringPrintf(" in %.3fs", timer.ElapsedSeconds());
}

// 这段代码实现了 SpatialPairGenerator 类的一个重载构造函数，
// 允许通过传递 SpatialMatchingOptions 和 Database 对象来初始化一个 SpatialPairGenerator 实例。
SpatialPairGenerator::SpatialPairGenerator(
    const SpatialMatchingOptions& options,
    const std::shared_ptr<Database>& database)
    : SpatialPairGenerator(
          options,
          std::make_shared<FeatureMatcherCache>(CacheSize(options),
                                                THROW_CHECK_NOTNULL(database),
                                                /*do_setup=*/true)) {}

void SpatialPairGenerator::Reset() { current_idx_ = 0; }

bool SpatialPairGenerator::HasFinished() const {
  return current_idx_ >= location_idxs_.size();
}

std::vector<std::pair<image_t, image_t>> SpatialPairGenerator::Next() {
  image_pairs_.clear();
  if (HasFinished()) {
    return image_pairs_;
  }

  LOG(INFO) << StringPrintf(
      "Matching image [%d/%d]", current_idx_ + 1, location_idxs_.size());
  const int knn =
      std::min<int>(options_.max_num_neighbors, location_idxs_.size());
  const float max_distance =
      static_cast<float>(options_.max_distance * options_.max_distance);
  for (int j = 0; j < knn; ++j) {
    // Check if query equals result.
    if (index_matrix_(current_idx_, j) == current_idx_) {
      continue;
    }

    // Since the nearest neighbors are sorted by distance, we can break.
    if (distance_matrix_(current_idx_, j) > max_distance) {
      break;
    }

    const image_t image_id = image_ids_.at(location_idxs_[current_idx_]);
    const size_t nn_idx = location_idxs_.at(index_matrix_(current_idx_, j));
    const image_t nn_image_id = image_ids_.at(nn_idx);
    image_pairs_.emplace_back(image_id, nn_image_id);
  }
  ++current_idx_;
  return image_pairs_;
}

Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>
// ReadLocationData 方法读取 FeatureMatcherCache 中的图像位置信息，并将其转换为一个矩阵 
// location_matrix，其中每一行代表一个图像的三维位置。该方法根据位姿先验数据的坐标系（WGS84或笛卡尔）进行相应的处理，并支持根据配置选项忽略Z坐标。
SpatialPairGenerator::ReadLocationData(const FeatureMatcherCache& cache) {
  GPSTransform gps_transform;
  std::vector<Eigen::Vector3d> ells(1);

  size_t num_locations = 0;
  location_idxs_.clear();
  location_idxs_.reserve(image_ids_.size());
  Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> location_matrix(
      image_ids_.size(), 3);

  for (size_t i = 0; i < image_ids_.size(); ++i) {
    if (!cache.ExistsPosePrior(image_ids_[i])) {
      continue;
    }
    const auto& pose_prior = cache.GetPosePrior(image_ids_[i]);
    const Eigen::Vector3d& translation_prior = pose_prior.position;
    if ((translation_prior(0) == 0 && translation_prior(1) == 0 &&
         options_.ignore_z) ||
        (translation_prior(0) == 0 && translation_prior(1) == 0 &&
         translation_prior(2) == 0 && !options_.ignore_z)) {
      continue;
    }

    location_idxs_.push_back(i);

    switch (pose_prior.coordinate_system) {
      case PosePrior::CoordinateSystem::WGS84: {
        ells[0](0) = translation_prior(0);
        ells[0](1) = translation_prior(1);
        ells[0](2) = options_.ignore_z ? 0 : translation_prior(2);

        const auto xyzs = gps_transform.EllToXYZ(ells);
        location_matrix(num_locations, 0) = static_cast<float>(xyzs[0](0));
        location_matrix(num_locations, 1) = static_cast<float>(xyzs[0](1));
        location_matrix(num_locations, 2) = static_cast<float>(xyzs[0](2));
      } break;
      case PosePrior::CoordinateSystem::UNDEFINED:
        LOG(INFO) << "Unknown coordinate system for image " << image_ids_[i]
                  << ", assuming cartesian.";
      case PosePrior::CoordinateSystem::CARTESIAN:
      default:
        location_matrix(num_locations, 0) =
            static_cast<float>(translation_prior(0));
        location_matrix(num_locations, 1) =
            static_cast<float>(translation_prior(1));
        location_matrix(num_locations, 2) =
            static_cast<float>(options_.ignore_z ? 0 : translation_prior(2));
    }

    num_locations += 1;
  }
  return location_matrix;
}

// 构造函数负责从一个外部文件中导入图像对，并将这些图像对存储在 image_pairs_ 中，以供后续的图像匹配过程使用。
ImportedPairGenerator::ImportedPairGenerator(
    const ImagePairsMatchingOptions& options,
    const std::shared_ptr<FeatureMatcherCache>& cache)
    : options_(options) {
  LOG(INFO) << "Importing image pairs...";
  THROW_CHECK(options.Check());

  const std::vector<image_t> image_ids = cache->GetImageIds();
  std::unordered_map<std::string, image_t> image_name_to_image_id;
  image_name_to_image_id.reserve(image_ids.size());
  for (const auto image_id : image_ids) {
    const auto& image = cache->GetImage(image_id);
    image_name_to_image_id.emplace(image.Name(), image_id);
  }
  image_pairs_ =
      ReadImagePairsText(options_.match_list_path, image_name_to_image_id);
  block_image_pairs_.reserve(options_.block_size);
}

ImportedPairGenerator::ImportedPairGenerator(
    const ImagePairsMatchingOptions& options,
    const std::shared_ptr<Database>& database)
    : ImportedPairGenerator(
          options,
          std::make_shared<FeatureMatcherCache>(CacheSize(options),
                                                THROW_CHECK_NOTNULL(database),
                                                /*do_setup=*/true)) {}

void ImportedPairGenerator::Reset() { pair_idx_ = 0; }

bool ImportedPairGenerator::HasFinished() const {
  return pair_idx_ >= image_pairs_.size();
}

std::vector<std::pair<image_t, image_t>> ImportedPairGenerator::Next() {
  block_image_pairs_.clear();
  if (HasFinished()) {
    return block_image_pairs_;
  }

  LOG(INFO) << StringPrintf("Matching block [%d/%d]",
                            pair_idx_ / options_.block_size + 1,
                            image_pairs_.size() / options_.block_size + 1);

  const size_t block_end =
      std::min(pair_idx_ + options_.block_size, image_pairs_.size());
  for (size_t j = pair_idx_; j < block_end; ++j) {
    block_image_pairs_.push_back(image_pairs_[j]);
  }
  pair_idx_ += options_.block_size;
  return block_image_pairs_;
}

}  // namespace colmap