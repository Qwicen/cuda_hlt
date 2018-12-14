#include "InputReader.h"

namespace {
   using std::make_pair;
}

Reader::Reader(const std::string& folder_name) : folder_name(folder_name) {
  if (!exists_test(folder_name)) {
    throw StrException("Folder " + folder_name + " does not exist.");
  }
}

std::vector<char> GeometryReader::read_geometry(const std::string& filename) const {
  std::vector<char> geometry;
  ::read_geometry(folder_name + "/" + filename, geometry);
  return geometry;
}

std::vector<char> UTMagnetToolReader::read_UT_magnet_tool() const {
  std::vector<char> ut_magnet_tool;
  ::read_UT_magnet_tool(folder_name, ut_magnet_tool);
  return ut_magnet_tool;
}

void EventReader::read_events(uint number_of_events_requested, uint start_event_offset) {

   for (auto bank_type : types()) {
      const auto& folder = this->folder(bank_type);

      std::vector<char> events;
      std::vector<uint> event_offsets;

      read_folder(folder,
                  number_of_events_requested,
                  events,
                  event_offsets,
                  start_event_offset);

      check_events(bank_type, events, event_offsets, number_of_events_requested);

      // TODO Remove: Temporal check to understand if number_of_events_requested is the same as number_of_events
      const int number_of_events = event_offsets.size() - 1;
      if (number_of_events_requested != number_of_events) {
         throw StrException("Number of events requested differs from number of events read.");
      }

      // Copy raw data to pinned host memory
      char* events_mem = nullptr;
      uint* offsets_mem = nullptr;
      cudaCheck(cudaMallocHost((void**)&events_mem, events.size()));
      cudaCheck(cudaMallocHost((void**)&offsets_mem, event_offsets.size() * sizeof(uint)));
      std::copy_n(std::begin(events), events.size(), events_mem);
      std::copy_n(std::begin(event_offsets), event_offsets.size(), offsets_mem);

      m_events[bank_type] = make_pair(gsl::span<char>{events_mem, events.size()},
                                      gsl::span<uint>{offsets_mem, event_offsets.size()});
   }
}

bool EventReader::check_events(BankTypes type,
                               const std::vector<char>& events,
                               const std::vector<uint>& event_offsets,
                               uint number_of_events_requested) const
{
   if (type == BankTypes::VP) {
      return check_velopix_events(events, event_offsets, number_of_events_requested);
   } else {
      return events.size() == event_offsets.back();
   }
}

CatboostModelReader::CatboostModelReader(const std::string& file_name) {
   if (!exists_test(file_name)) {
      throw StrException("Catboost model file " + file_name + " does not exist.");
   }
   std::ifstream i(file_name);
   nlohmann::json j;
   i >> j;
   m_num_features = j["features_info"]["float_features"].size();
   m_num_trees = j["oblivious_trees"].size();
   m_tree_offsets.push_back(0);
   m_leaf_offsets.push_back(0);
   for (nlohmann::json::iterator it = j["oblivious_trees"].begin(); it != j["oblivious_trees"].end(); ++it) {
      nlohmann::json tree(*it); 
      std::vector<float> tree_split_borders;
      std::vector<int> tree_split_features;
      m_leaf_values.insert(std::end(m_leaf_values), std::begin(tree["leaf_values"]), std::end(tree["leaf_values"]));
      m_tree_depths.push_back(tree["splits"].size());
      m_tree_offsets.push_back(m_tree_offsets.back() + m_tree_depths.back());
      m_leaf_offsets.push_back(m_leaf_offsets.back() + (1<<m_tree_depths.back()));
      for(nlohmann::json::iterator it_spl = tree["splits"].begin(); it_spl != tree["splits"].end(); ++it_spl) {
         nlohmann::json split(*it_spl);
         tree_split_borders.push_back(split["border"]);
         tree_split_features.push_back(split["float_feature_index"]);
      }
      m_split_border.insert(std::end(m_split_border), std::begin(tree_split_borders), std::end(tree_split_borders));
      m_split_feature.insert(std::end(m_split_feature), std::begin(tree_split_features), std::end(tree_split_features));
   }
}