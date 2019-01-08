#include "CheckerInvoker.h"

std::tuple<bool, MCEvents> CheckerInvoker::read_mc_folder() const
{
  std::string mc_tracks_folder = mc_folder + "/tracks";
  const auto folder_file_list = list_folder(mc_tracks_folder);

  uint requested_files =
      number_of_requested_events == 0 ? folder_file_list.size() : number_of_requested_events;
  verbose_cout << "Requested " << requested_files << " files" << std::endl;

  if (requested_files > folder_file_list.size()) {
    error_cout << "Monte Carlo validation failed: Requested " << requested_files
               << " events, but only " << folder_file_list.size()
               << " Monte Carlo files are present." << std::endl
               << std::endl;

    return {false, {}};
  }

  std::vector<MCEvent> input;
  int readFiles = 0;
  for (uint i = start_event_offset; i < requested_files + start_event_offset;
       ++i) {
    // Read event #i in the list and add it to the inputs
    std::string readingFile = folder_file_list[i];

    std::vector<char> input_contents;
    readFileIntoVector(mc_tracks_folder + "/" + readingFile, input_contents);
    const auto event = MCEvent(input_contents, check_events);

    // debug_cout << "At MCEvent " << i << ": " << int(event.mcps.size())
    // << " MCPs" << std::endl;
    // if ( i == 0 && check_events )
    //      event.print();

    input.emplace_back(event);

    readFiles++;
    if ((readFiles % 100) == 0) {
      info_cout << "." << std::flush;
    }
  }

  info_cout << std::endl
            << input.size() << " files read" << std::endl
            << std::endl;
  return {true, input};
}
