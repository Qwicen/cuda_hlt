#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <unordered_set>
#include <map>

#include "Tools.h"
#include "InputTools.h"

namespace {
  using std::cout;
  using std::endl;
  using std::ifstream;
  using std::ios;
  using std::make_pair;
  using std::map;
  using std::string;
  using std::to_string;
  using std::unordered_set;
  using std::vector;
} // namespace

int main(int argc, char* argv[])
{
  if (argc <= 1) {
    cout << "usage: test_read <file.mdf>" << endl;
    return -1;
  }

  string filename = {argv[1]};
  vector<char> data;
  vector<unsigned int> offsets(1, 0);
  appendFileToVector(filename, data, offsets);

  check_velopix_events(data, offsets, 1);

  for (size_t i = 0; i < offsets.size(); ++i) {
    cout << offsets[i] << endl;
  }
}
