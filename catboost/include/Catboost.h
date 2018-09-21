#pragma once

#include "evaluator.h"

class CatboostEvaluator : public NCatboostStandalone::TZeroCopyEvaluator {
  CatboostEvaluator() = delete;
public:
  CatboostEvaluator(const std::string& modelFile);
  CatboostEvaluator(const std::vector<unsigned char>& modelBlob);
  CatboostEvaluator(std::vector<unsigned char>&& modelBlob);

  int GetBinFeatureCount() const {
    return BinaryFeatureCount;
  }

  const NCatBoostFbs::TObliviousTrees* GetObliviousTrees() const {
    return ObliviousTrees;
  }
private:
  void InitEvaluator();
private:
  std::vector<unsigned char> ModelBlob;
  const NCatBoostFbs::TObliviousTrees* ObliviousTrees = nullptr;
  size_t BinaryFeatureCount = 0;
};