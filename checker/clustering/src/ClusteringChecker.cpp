#include "../include/ClusteringChecker.h"

/**
 * @brief A utility to check the efficiency of a clustering algorithm.
 * @details The input are the geometry, events and event_offsets
 *          as read from the input files (raw events), and the
 *          found_clusters as a vector of LHCb IDs for each event
 *          (hence vector of vector of LHCb ID).
 *
 *          The output is three floats:
 *
 *          - Reconstruction efficiency:
 *            Number of correctly reconstructed clusters / Total
 *            number of reconstructible clusters
 *
 *          - Clone fraction:
 *            Number of clone clusters / Number of reconstructed clusters
 *
 *          - Ghost fraction:
 *            Number of ghost clusters / Number of reconstructed clusters
 *
 *          Parameters to control the clustering checker:
 *
 *          - just_check_ids: Performs just a mere LHCb ID comparison.
 *
 *          - allowed_distance_error: If just_check_ids is false, for each LHCb ID
 *            it retrieves the row and column, and for each cluster it checks that it
 *            is at most at an allowed_distance_error from a reconstructible cluster.
 *
 * @return tuple <re, cf, gf>
 */
void checkClustering(
  const std::vector<char>& geometry,
  const std::vector<char>& events,
  const std::vector<uint>& event_offsets,
  const std::vector<std::vector<uint32_t>>& found_clusters,
  float& reconstruction_efficiency,
  float& clone_fraction,
  float& ghost_fraction,
  const bool just_check_ids,
  const float allowed_distance_error)
{
  std::vector<std::vector<uint32_t>> expected_clusters = clustering(geometry, events, event_offsets);

  reconstruction_efficiency = 0.f;
  clone_fraction = 0.f;
  ghost_fraction = 0.f;

  if (found_clusters.size() != expected_clusters.size()) {
    warning_cout << "Cluster checker: Found cluster size " << found_clusters.size()
                 << " does not match expected cluster size " << expected_clusters.size() << "." << std::endl;
    return;
  }

  const auto number_of_events = found_clusters.size();
  uint total_number_of_reconstructible_clusters = 0;
  uint number_of_reconstructed_clusters = 0;
  uint number_of_correctly_reconstructed_clusters = 0;
  uint number_of_clones = 0;
  uint number_of_ghosts = 0;

  for (uint i = 0; i < number_of_events; ++i) {
    total_number_of_reconstructible_clusters += expected_clusters[i].size();
  }

  for (uint i = 0; i < number_of_events; ++i) {
    number_of_reconstructed_clusters += found_clusters[i].size();

    for (auto lhcb_id : found_clusters[i]) {
      const uint row = lhcb_id & 0xFF;
      const uint col = (lhcb_id >> 8) & 0xFF;
      const uint det_sensor_chip = lhcb_id & 0xFFFF0000;
      if (
        std::find_if(
          expected_clusters[i].begin(),
          expected_clusters[i].end(),
          [&just_check_ids, &row, &col, &allowed_distance_error, &lhcb_id, &det_sensor_chip](uint32_t other_lhcb_id) {
            if (lhcb_id == other_lhcb_id) {
              return true;
            }
            else if (!just_check_ids && det_sensor_chip == (other_lhcb_id & 0xFFFF0000)) {
              const uint other_row = other_lhcb_id & 0xFF;
              const uint other_col = (other_lhcb_id >> 8) & 0xFF;
              const float distance = (row - other_row) * (row - other_row) + (col - other_col) * (col - other_col);
              if (std::sqrt(distance) < allowed_distance_error) {
                return true;
              }
            }
            return false;
          }) != expected_clusters[i].end()) {
        ++number_of_correctly_reconstructed_clusters;
      }
      else {
        ++number_of_ghosts;
      }
    }
    // Find clones: First create a set out of the vector
    std::set<uint32_t> s(found_clusters[i].begin(), found_clusters[i].end());
    for (auto lhcb_id : s) {
      const uint row = lhcb_id & 0xFF;
      const uint col = (lhcb_id >> 8) & 0xFF;
      const uint det_sensor_chip = lhcb_id & 0xFFFF0000;
      const auto cluster_repetitions = std::count_if(
        found_clusters[i].begin(),
        found_clusters[i].end(),
        [&just_check_ids, &row, &col, &allowed_distance_error, &lhcb_id, &det_sensor_chip](uint32_t other_lhcb_id) {
          if (lhcb_id == other_lhcb_id) {
            return true;
          }
          else if (!just_check_ids && det_sensor_chip == (other_lhcb_id & 0xFFFF0000)) {
            const uint other_row = other_lhcb_id & 0xFF;
            const uint other_col = (other_lhcb_id >> 8) & 0xFF;
            const float distance = (row - other_row) * (row - other_row) + (col - other_col) * (col - other_col);
            if (std::sqrt(distance) < allowed_distance_error) {
              return true;
            }
          }
          return false;
        });
      if (cluster_repetitions > 1) {
        number_of_clones += cluster_repetitions - 1;
      }
    }
  }

  if (total_number_of_reconstructible_clusters > 0) {
    reconstruction_efficiency =
      ((float) number_of_correctly_reconstructed_clusters) / ((float) total_number_of_reconstructible_clusters);
  }

  if (number_of_reconstructed_clusters > 0) {
    clone_fraction = ((float) number_of_clones) / ((float) number_of_reconstructed_clusters);
    ghost_fraction = ((float) number_of_ghosts) / ((float) number_of_reconstructed_clusters);
  }
}
