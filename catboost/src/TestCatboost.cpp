#include "TestCatboost.h"

void test_cpu_catboost_evaluator(
  const std::string& model_path,
  const std::string& data_path
) {
  NCatboostStandalone::TOwningEvaluator evaluator(model_path);
  std::ifstream file(data_path);
  std::vector<float> features;
  std::string line;
  std::string cell;

  int index = 0;
  float result = 0;
  while( file ) {
    index++;
    std::getline(file,line);
    std::stringstream lineStream(line);
    features.clear();

    while( std::getline( lineStream, cell, ',' ) )
      features.push_back( std::stof(cell) );
      
    result += evaluator.Apply(features, NCatboostStandalone::EPredictionType::Probability);
  }
  std::cout << data_path << ": " << result / index << std::endl;
}