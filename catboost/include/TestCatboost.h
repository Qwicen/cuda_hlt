#include "Catboost.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

void read_data(
  const std::string& data_path,
  std::vector<std::vector<float>>& features
);

void test_cpu_catboost_evaluator(
  const std::string& model_path,
  std::vector<std::vector<float>>& features
);