#include <list>
#include <algorithm>
#include "../../../main/include/Common.h"
#include "../../../main/include/Logger.h"

struct MemoryManager {
  constexpr static size_t max_available_memory = (size_t) 8 * 1024 * 1024 * 1024; // 8 GiB
  constexpr static uint guarantee_alignment = 256;

  /**
   * @brief A memory segment is composed of a start
   *        and size, both referencing bytes.
   *        The tag can either be -1 (free), or i \in {0-n},
   *        which means it is occupied by argument i
   */
  struct MemorySegment {
    uint start;
    size_t size;
    int tag;
  };

  std::list<MemorySegment> memory_segments {{0, max_available_memory, -1}};
  size_t total_memory_required = 0;

  MemoryManager() = default;

  /**
   * @brief Reserves a memory request of size requested_size.
   *        It can be overriden by other memory managers. The
   *        base version finds the first available segment. If
   *        there are no available segments of the requested size,
   *        it throws an exception.
   */
  virtual uint reserve(int tag, size_t requested_size) {
    // Aligned requested size
    auto aligned_request = requested_size + guarantee_alignment - 1
      - ((requested_size + guarantee_alignment - 1) % guarantee_alignment);

    debug_cout << "MemoryManager: Requested "
      << requested_size << " B (" << aligned_request << " B aligned)" << std::endl;

    auto it = memory_segments.begin();
    for (; it!=memory_segments.end(); ++it) {
      if (it->tag == -1 && it->size >= aligned_request) {
        break;
      }
    }
    if (it == memory_segments.end()) {
      throw StrException("Reserve: Requested size could not be met ("
        + std::to_string(aligned_request) + " B)");
    }

    // Start of allocation
    uint start = it->start;

    // Update current segment
    it->start += aligned_request;
    it->size -= aligned_request;
    if (it->size == 0) {
      it = memory_segments.erase(it);
    }

    // Insert an occupied segment
    auto segment = MemorySegment{start, aligned_request, tag};
    memory_segments.insert(it, segment);

    // Update total memory required
    // Note: This can be done accesing the last element in memory_segments
    //       upon every reserve, and keeping the maximum used memory
    total_memory_required = std::max(total_memory_required,
      max_available_memory - memory_segments.back().size);

    return start;
  }

  /**
   * @brief Frees the memory segment occupied by the tag.
   */
  virtual void free(int tag) {
    auto it = std::find_if(memory_segments.begin(), memory_segments.end(),
      [&tag] (const MemorySegment& segment) {
        if (segment.tag == tag) {
          return true;
        }
        return false;
    });

    if (it == memory_segments.end()) {
      throw StrException("Free: Requested tag could not be found ("
        + std::to_string(tag) + ")");
    }

    // Free found tag
    it->tag = -1;

    // Check if previous segment is free, in which case, join
    if (it != memory_segments.begin()) {
      auto previous_it = std::prev(it);
      if (previous_it->tag == -1) {
        previous_it->size += it->size;
        // Remove current element, and point to previous one
        it = memory_segments.erase(it);
        it--;
      }
    }

    // Check if next segment is free, in which case, join
    if (std::next(it) != memory_segments.end()) {
      auto next_it = std::next(it);
      if (next_it->tag == -1) {
        it->size += next_it->size;
        // Remove next tag
        memory_segments.erase(next_it);
      }
    }
  }

  /**
   * @brief Frees all memory segments, effectively resetting the
   *        available space.
   */
  void free_all() {
    memory_segments = std::list<MemorySegment>{{0, max_available_memory, -1}};
  }

  /**
   * @brief Prints the current state of the memory segments.
   */
  void print(const std::vector<std::string>& argument_names, const int step = -1) {
    info_cout << "Memory Manager max memory required (MiB): "
      << (total_memory_required) << std::endl;

    if (step!=-1) { info_cout << "Sequence step " << step << " memory segments (MiB): "; }
    else { info_cout << "MemoryManager segments (MiB): "; }

    for (auto& segment : memory_segments) {
      std::string name = segment.tag==-1 ? "unused" : argument_names[segment.tag];
      info_cout << name << " (" << ((float) segment.size) / (1024 * 1024) << "), ";
    }
    info_cout << std::endl;
  }
};
