#include "Catboost.h"
#include "evaluator.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

std::vector<std::vector<float>> read_csv_data_file(
  const std::string& data_path
);

void test_cpu_catboost_evaluator(
  const std::string& model_path,
  std::vector<std::vector<float>>& features
);