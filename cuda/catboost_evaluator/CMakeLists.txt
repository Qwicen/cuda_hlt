file(GLOB catboost_evaluator "evaluator/src/*cu")
include_directories(evaluator/include)

cuda_add_library(Evaluator STATIC
  ${catboost_evaluator}
)