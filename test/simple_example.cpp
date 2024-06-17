#include <bvh/v2/bvh.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/executor.h>
#include <bvh/v2/node.h>
#include <bvh/v2/ray.h>
#include <bvh/v2/stack.h>
#include <bvh/v2/thread_pool.h>
#include <bvh/v2/tri.h>
#include <bvh/v2/vec.h>

#include <iostream>

using Scalar = float;
using Vec3 = bvh::v2::Vec<Scalar, 3>;
using BBox = bvh::v2::BBox<Scalar, 3>;
using Tri = bvh::v2::Tri<Scalar, 3>;
using Node = bvh::v2::Node<Scalar, 3>;
using Bvh = bvh::v2::Bvh<Node>;
using Ray = bvh::v2::Ray<Scalar, 3>;

using PrecomputedTri = bvh::v2::PrecomputedTri<Scalar>;

int main() {
  // This is the original data, which may come in some other data
  // type/structure.
  std::vector<Tri> tris;
  tris.emplace_back(Vec3(1.0, -1.0, 1.0), Vec3(1.0, 1.0, 1.0),
                    Vec3(2.0, 1.0, 1.0));
  tris.emplace_back(Vec3(1.0, -1.0, 1.0), Vec3(-1.0, -1.0, 1.0),
                    Vec3(-1.0, 1.0, 1.0));

  bvh::v2::ThreadPool thread_pool;
  bvh::v2::ParallelExecutor executor(thread_pool);

  // Get triangle centers and bounding boxes (required for BVH builder)
  std::vector<BBox> bboxes(tris.size());
  std::vector<Vec3> centers(tris.size());
  executor.for_each(0, tris.size(), [&](size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
      bboxes[i] = tris[i].get_bbox();
      centers[i] = tris[i].get_center();
    }
  });

  printf("boxes:\n");
  for(int i = 0; i < bboxes.size(); ++i) {
    printf("i: %d\n", i);
    printf("min: %f %f %f\n", bboxes[i].min[0], bboxes[i].min[1], bboxes[i].min[2]);
    printf("max: %f %f %f\n", bboxes[i].max[0], bboxes[i].max[1], bboxes[i].max[2]);
  }
  printf("-----\n");

  typename bvh::v2::DefaultBuilder<Node>::Config config;
  config.quality = bvh::v2::DefaultBuilder<Node>::Quality::Low;
  auto bvh = bvh::v2::DefaultBuilder<Node>::build(thread_pool, bboxes, centers,
                                                  config);

  // Permuting the primitive data allows to remove indirections during
  // traversal, which makes it faster.
  static constexpr bool should_permute = true;

  // This precomputes some data to speed up traversal further.
  std::vector<PrecomputedTri> precomputed_tris(tris.size());
  executor.for_each(0, tris.size(), [&](size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
      auto j = should_permute ? bvh.prim_ids[i] : i;
      precomputed_tris[i] = tris[j];
    }
  });

  auto ray = Ray{
      Vec3(0., 0., 0.), // Ray origin
      Vec3(0., 0., 1.), // Ray direction
      0.,               // Minimum intersection distance
      100.              // Maximum intersection distance
  };

  static constexpr size_t invalid_id = std::numeric_limits<size_t>::max();
  static constexpr size_t stack_size = 64;
  static constexpr bool use_robust_traversal = false;

  auto prim_id = invalid_id;
  Scalar u, v;

  if (false) {
    // Traverse the BVH and get the u, v coordinates of the closest
    // intersection.
    bvh::v2::SmallStack<Bvh::Index, stack_size> stack;
    bvh.intersect<false, use_robust_traversal>(
        ray, bvh.get_root().index, stack, [&](size_t begin, size_t end) {
          for (size_t i = begin; i < end; ++i) {
            size_t j = should_permute ? i : bvh.prim_ids[i];
            if (auto hit = precomputed_tris[j].intersect(ray)) {
              prim_id = i;
              std::tie(u, v) = *hit;
            }
          }
          return prim_id != invalid_id;
        });

    if (prim_id != invalid_id) {
      std::cout << "Intersection found\n"
                << "  primitive: " << prim_id << "\n"
                << "  distance: " << ray.tmax << "\n"
                << "  barycentric coords.: " << u << ", " << v << std::endl;
      return 0;
    } else {
      std::cout << "No intersection found" << std::endl;
      return 1;
    }
  }

  {
    bvh::v2::SmallStack<Bvh::Index, stack_size> stack2;
    
    BBox bbox;
    bbox.min = Vec3(0.6, 0, 1.0);
    bbox.max = Vec3(0.7, 0.1, 1.0);
    auto box_box_intersection = [&](
        const Vec3& min1,
        const Vec3& max1,
      const Vec3& min2,
      const Vec3& max2) -> bool 

    {
        if (max1[0] < min2[0] || max1[1] < min2[1] || max1[2] < min2[2])
            return 0;
        if (max2[0] < min1[0] || max2[1] < min1[1] || max2[2] < min1[2])
            return 0;
        return 1;
    };

    auto inner_fn = [&](const Node &left, const Node &right) {
      printf("left %d right %d\n", left.index.first_id(), right.index.first_id());
      auto left_bbox = left.get_bbox();
      auto left_min = left_bbox.min;
      auto left_max = left_bbox.max;
      auto left_inters = box_box_intersection(left_min, left_max, bbox.min, bbox.max);
      auto right_bbox = right.get_bbox();
      auto right_min = right_bbox.min;
      auto right_max = right_bbox.max;
      auto right_inters = box_box_intersection(right_min, right_max, bbox.min, bbox.max);
      printf("left_inters: %d right_inters: %d\n", left_inters, right_inters);

      return std::array<bool, 3>({left_inters, right_inters, false});
    };

    std::vector<int> tmp;
    auto leaf_fn = [&](Bvh::Index::Type st, Bvh::Index::Type end) {
      printf("leaf_fn %d %d\n", st, end);

      for(Bvh::Index::Type i = st; i < end; ++i) {
        printf("i: %d\n", i);
        auto node = bvh.nodes[i];
        printf("prim_ids[]: %d\n", bvh.prim_ids[i]);
        // printf("prim_counts: %d\n", node.index.prim_count());
        auto min_bb = node.get_bbox().min;
        auto max_bb = node.get_bbox().max;
        printf("min: %f %f %f\n", min_bb[0], min_bb[1], min_bb[2]);
        printf("max: %f %f %f\n", max_bb[0], max_bb[1], max_bb[2]);
        auto inters = box_box_intersection(min_bb, max_bb, bbox.min, bbox.max);
        printf("i :%d, inters %d\n",i, inters);
        if (inters) {
          tmp.push_back(i);
        }
      }
      return false;
    };

    const auto& root = bvh.get_root();
    printf("root is leaf: %d\n", root.index.is_leaf());
    printf("root.index.prim_counts: %d\n", root.index.prim_count());
    auto root_bbox = root.get_bbox();
    auto min_bb = root_bbox.min;
    auto max_bb = root_bbox.max;
    printf("min: %f %f %f\n", min_bb[0], min_bb[1], min_bb[2]);
    printf("max: %f %f %f\n", max_bb[0], max_bb[1], max_bb[2]);

    auto prim_size = bvh.prim_ids.size();
    printf("prim_size: %d\n", prim_size);
    // bvh::v2::
    // bvh.traverse_top_down<false>(root.index, stack2, leaf_fn, inner_fn);
    // Ray ray2;
    // ray2.org = Vec3(0.6, 0, 1.0);
    // ray2.dir = Vec3(0.7, 0.1, 1.0);

    float _min[3] = {0.6, 0, 1.0};
    float _max[3] = {0.7, 0.1, 1.0};

    bvh.intersect_box<false>(_min, _max, stack2, tmp);
    printf("tmp: \n");
    for(auto i : tmp) {
      printf("i: %d\n", i);
    }
  }
}
