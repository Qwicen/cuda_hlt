#include "catch.hpp"
#include "evaluator.h"
#include "MuonDefinitions.cuh"
#include "InputReader.h"
#include "Constants.cuh"
#include "MuonCatboostKernel.cuh"
#include <iostream>
#include <vector>

SCENARIO("Compare output of vanila catboost evaluator and GPU evaluator")
{
    GIVEN("Models & Batch of real data")
    {
        NCatboostStandalone::TOwningEvaluator evaluator("../input/model.bin");
        auto modelFloatFeatureCount = (size_t)evaluator.GetFloatFeatureCount();

        std::unique_ptr<CatboostModelReader> muon_catboost_model_reader;
        muon_catboost_model_reader = std::make_unique<CatboostModelReader>("../input/muon/muon_catboost_model.json");
        Constants constants;
        constants.initialize_muon_catboost_model_constants(
            muon_catboost_model_reader->n_features(),
            muon_catboost_model_reader->n_trees(),
            muon_catboost_model_reader->tree_depths(),
            muon_catboost_model_reader->tree_offsets(),
            muon_catboost_model_reader->leaf_values(),
            muon_catboost_model_reader->leaf_offsets(),
            muon_catboost_model_reader->split_border(),
            muon_catboost_model_reader->split_feature());

        std::vector<std::vector<float>> features = {
            {2, 2, 1, 1, 5 , 3, 7 , 3,  1,  0, 7 , 3 , -1.0408108, -0.7461224, -0.75673074,	-0.8203014, -1.0784016,  -0.8081739,  -0.5914081, -0.4181551 },
            {2, 2, 2, 2, 13, 8, 5 , 10, 3,  0, -1, -1, -1.8238717, -1.5396439, -0.48220623,	0.058582287, 1.9964803,  -0.9753074, -0.86314434, -0.73493594},
            {2, 2, 1, 1, 7 , 6, 11, 5,  0,  0, 11, 5 , 0.38115135, -0.21937361, -0.1941489,	0.3990361,   0.6911263,     0.57658,  0.46513978, -1.6382334 },
            {2, 2, 1, 1, 5 , 7, 6 , 8, -1, -2, 6 , 8 , 1.4522418,  1.7589493,    1.8225902,	2.0780447, -0.75160056, -0.60858285, -0.49724054, -0.42617154},
            {2, 2, 1, 1, 8 , 7, 4 , 10, 4, -6, 4 , 10, -2.242363,  -2.5331066,  -2.6107218,	-2.6248055, -1.4353153,  -0.9037572,  -1.2505248, -0.9549604 }
        };

        const int n_tracks = features.size();

        WHEN("Both models are evaluated")
        {
            std::vector<float> output_vanila_cb(n_tracks);
            std::vector<float> output_gpu_cb;

            for (size_t i = 0; i < n_tracks; ++i) {
                output_vanila_cb[i] = evaluator.Apply(features[i], NCatboostStandalone::EPredictionType::RawValue);
            }

            output_gpu_cb = run_kernel(features, constants);

            THEN("Their output is close (within 1e-5)")
            {
                const float eps = 1e-5;
                for (int i = 0; i < n_tracks; i++) {
                    CHECK_THAT(
                        output_vanila_cb[i],
                        Catch::Matchers::WithinAbs(output_gpu_cb[i], eps)
                    );
                }
            }
        }
    }
}