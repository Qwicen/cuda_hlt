#pragma once

#include "model_generated.h"

#include <string>
#include <vector>

namespace NCatboostStandalone {
    enum class EPredictionType {
        //! Just raw sum of leaf values of model trees
        RawValue,
        //! Apply sigmoid to raw sum of leaf values to evaluate probability
        Probability,
        //! Get class prediction (if raw value is greater than zero return 1, else 0)
        Class
    };

    class TZeroCopyEvaluator {
    public:
        TZeroCopyEvaluator() = default;

        TZeroCopyEvaluator(const NCatBoostFbs::TModelCore* core);

        double Apply(const std::vector<float>& features, EPredictionType predictionType) const;

        void SetModelPtr(const NCatBoostFbs::TModelCore* core);

        int GetFloatFeatureCount() const {
            return FloatFeatureCount;
        }

        int GetBinFeatureCount() const {
            return BinaryFeatureCount;
        }

        const NCatBoostFbs::TObliviousTrees* GetObliviousTrees() const {
            return ObliviousTrees;
        }
    private:
        const NCatBoostFbs::TObliviousTrees* ObliviousTrees = nullptr;
        size_t BinaryFeatureCount = 0;
        int FloatFeatureCount = 0;
    };

    class TOwningEvaluator : public TZeroCopyEvaluator {
        TOwningEvaluator() = delete;
    public:
        TOwningEvaluator(const std::string& modelFile);
        TOwningEvaluator(const std::vector<unsigned char>& modelBlob);
        TOwningEvaluator(std::vector<unsigned char>&& modelBlob);
    private:
        void InitEvaluator();
    private:
        std::vector<unsigned char> ModelBlob;
    };
}
