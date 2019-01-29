#include "catch.hpp"
#include "../main/include/Tools.h"

SCENARIO("General case") { 
  const std::string MUON_COORDS_FOLDER = "input/minbias/muon_coords";
  const std::string MUON_COMMON_HITS_FOLDER = "input/minbias/muon_common_hits";
  const std::vector<const std::string> DATA_FILES { 
    "6718861_6001.bin", 
    "6718861_6002.bin", 
    "6718861_6003.bin", 
    "6718861_6004.bin", 
    "6718861_6005.bin", 
    "6718861_6006.bin", 
    "6718861_6007.bin", 
    "6718861_6008.bin", 
    "6718861_6009.bin", 
    "6718861_6010.bin" 
  };
  const int N_EVENTS = DATA_FILES.size();
  const uint MUON_COORDS_EVENT_OFFSETS[] = { 10796, 920, 4120, 5904, 4620, 1636, 5140, 2928, 5860, 7320 };
  const uint MUON_COMMON_HITS_EVENT_OFFSETS[] = { 24128, 2128, 9256, 13348, 10400, 3756, 11500, 6748, 13172, 16648 };
  for (size_t i = 0; i < N_EVENTS; i++) {
    const auto& data_file = DATA_FILES[i];
    ifstream muon_coords_file(MUON_COORDS_FOLDER + "/" + data_file, ios::in | ios::binary);
    char muon_coords_raw_input[MUON_COORDS_EVENT_OFFSETS[i]];
    muon_coords_file.read(muon_coords_raw_input, MUON_COORDS_EVENT_OFFSETS[i]);
    muon_coords_file.close();
    Muon::MuonCoords muon_coords;
	read_muon_coords(&muon_coords, muon_coords_raw_input);

    ifstream muon_common_hits_file(MUON_COMMON_HITS_FOLDER + "/" + data_file, ios::in | ios::binary);
    char muon_common_hits_raw_input[MUON_COMMON_HITS_EVENT_OFFSETS[i]];
    muon_common_hits_file.read(muon_common_hits_raw_input, MUON_COMMON_HITS_EVENT_OFFSETS[i]);
    muon_common_hits_file.close();
    Muon::HitsSoA muon_common_hits;
	read_muon_events_into_arrays(&muon_common_hits, muon_common_hits_raw_input, {0}, 1);

	Muon::HitsSoA transformed;
    transform_muon_coords_to_muon_common_hits(&muon_coords, &transformed);
    REQUIRE(muon_coords == transformed);
  }
}
