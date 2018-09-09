#include "TestCatboost.h"

void test_cpu_catboost_evaluator(
  const std::string& model_path,
  std::vector<std::vector<float>>& features
) {
  NCatboostStandalone::TOwningEvaluator evaluator(model_path);
  int index = 0;
  float result = 0;
  for( const std::vector<float> event : features ) {
    index++;
    result += evaluator.Apply(event, NCatboostStandalone::EPredictionType::Probability);
  }
  std::cout << "result: " << result / index << std::endl;
}

void read_data(
  const std::string& data_path,
  std::vector<std::vector<float>>& features
) {
  std::ifstream file(data_path);
  std::vector<float> event;
  std::string line;
  std::string cell;

  while( file ) {
    std::getline(file,line);
    std::stringstream lineStream(line);
    event.clear();

    while( std::getline( lineStream, cell, ',' ) )
      event.push_back( std::stof(cell) );
    if(!event.empty())
      features.push_back(event);
  }
}