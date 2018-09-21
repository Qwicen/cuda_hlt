#include "Catboost.h"

#include <algorithm>
#include <exception>
#include <fstream>
#include <functional>


static const char MODEL_FILE_DESCRIPTOR_CHARS[4] = {'C', 'B', 'M', '1'};

namespace {
  unsigned int GetModelFormatDescriptor() {
    static_assert(sizeof(unsigned int) == 4, "");
    unsigned int result;
    memcpy(&result, MODEL_FILE_DESCRIPTOR_CHARS, sizeof(unsigned int));
    return result;
  }
}

CatboostEvaluator::CatboostEvaluator(const std::string& modelFile) {
  std::ifstream file(modelFile, std::ios::binary);
  ModelBlob.clear();
  ModelBlob.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  InitEvaluator();
}

CatboostEvaluator::CatboostEvaluator(std::vector<unsigned char>&& modelBlob) {
  ModelBlob = std::move(modelBlob);
  InitEvaluator();
}


CatboostEvaluator::CatboostEvaluator(const std::vector<unsigned char>& modelBlob) {
  ModelBlob = modelBlob;
  InitEvaluator();
}

void CatboostEvaluator::InitEvaluator() {
  const auto modelBufferStartOffset = sizeof(unsigned int) * 2;
  if (ModelBlob.empty()) {
    throw std::runtime_error("trying to initialize evaluator from empty ModelBlob");
  }
  {
    const unsigned int* intPtr = reinterpret_cast<const unsigned int*>(ModelBlob.data());
    // verify model file descriptor
    if (intPtr[0] != GetModelFormatDescriptor()) {
      throw std::runtime_error("incorrect model format descriptor");
    }
    // verify model blob length
    if (intPtr[1] + modelBufferStartOffset > ModelBlob.size()) {
      throw std::runtime_error("insufficient model length");
    }
  }
  auto flatbufStartPtr = ModelBlob.data() + modelBufferStartOffset;
  // verify flatbuffers
  {
    flatbuffers::Verifier verifier(flatbufStartPtr, ModelBlob.size() - modelBufferStartOffset);
    if (!NCatBoostFbs::VerifyTModelCoreBuffer(verifier)) {
      throw std::runtime_error("corrupted flatbuffer model");
    }
  }
  auto flatbufModelCore = NCatBoostFbs::GetTModelCore(flatbufStartPtr);
  SetModelPtr(flatbufModelCore);
  ObliviousTrees = flatbufModelCore->ObliviousTrees();
}
