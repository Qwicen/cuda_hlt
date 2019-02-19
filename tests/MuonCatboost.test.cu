#include "catch.hpp"
#include "Catboost.h"
#include "MuonDefinitions.cuh"
#include "MuonCatboostEvaluator.cuh"

SCENARIO("lol")
{
    GIVEN("KEK")
    {
        NCatboostStandalone::TOwningEvaluator evaluator("../model.bin");
        
        WHEN("cheburek")
        {
            muon_catboost_evaluator<<<dim3(1), dim3(1)>>>(
                dev_muon_catboost_features,
                dev_muon_catboost_output,
                dev_muon_catboost_leaf_values,
                dev_muon_catboost_leaf_offsets,
                dev_muon_catboost_split_borders,
                dev_muon_catboost_split_features,
                dev_muon_catboost_tree_sizes,
                dev_muon_catboost_tree_offsets,
                n_trees);

            std::vector<float> output(host_buffers.host_number_of_reconstructed_scifi_tracks[0]);
            
            cudaCheck(cudaMemcpyAsync(
              output.data(),
              arguments.offset<dev_muon_catboost_output>(),
              arguments.size<dev_muon_catboost_output>(),
              cudaMemcpyDeviceToHost,
              cuda_stream));

            THEN("qwert")
            {
                //output == ...
                CHECK(5 == 5);
            }
        }
    }
}