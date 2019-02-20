#include "MuonCatboostKernel.cuh"
#include "MuonCatboostEvaluator.cuh"

std::vector<float> run_kernel(std::vector<std::vector<float>>& features, Constants& constants) {
    float* dev_muon_catboost_features;
    float* dev_muon_catboost_output;
    const int n_tracks = features.size();

    cudaMalloc(&dev_muon_catboost_features, 
        n_tracks * constants.muon_catboost_n_features * sizeof(float));
    cudaMalloc(&dev_muon_catboost_output, n_tracks * sizeof(float));

    cudaMemcpy(
        dev_muon_catboost_features,
        flatten(features).data(),
        n_tracks * constants.muon_catboost_n_features * sizeof(float),
        cudaMemcpyHostToDevice
    );

    muon_catboost_evaluator<<<dim3(n_tracks), dim3(32)>>>(
        dev_muon_catboost_features,
        dev_muon_catboost_output,
        constants.dev_muon_catboost_leaf_values,
        constants.dev_muon_catboost_leaf_offsets,
        constants.dev_muon_catboost_split_borders,
        constants.dev_muon_catboost_split_features,
        constants.dev_muon_catboost_tree_depths,
        constants.dev_muon_catboost_tree_offsets,
        constants.muon_catboost_n_trees);

    std::vector<float> output(n_tracks);
    cudaMemcpy(output.data(), dev_muon_catboost_output, n_tracks * sizeof(float), cudaMemcpyDeviceToHost);

    return output;
}