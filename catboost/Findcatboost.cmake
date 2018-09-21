find_path(catboost_INCLUDE_FBS NAMES model.fbs
          HINTS ${catboost_home}/catboost/libs/model/flatbuffers)

find_path(catboost_INCLUDE_DIR NAMES evaluator.h
          HINTS ${catboost_home}/catboost/libs/standalone_evaluator)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(catboost 
  DEFAULT_MSG catboost_INCLUDE_FBS catboost_INCLUDE_DIR)

if(catboost_FOUND)
  set(catboost_FBS ${catboost_INCLUDE_FBS})
  set(catboost_INCLUDE_DIRS ${catboost_INCLUDE_DIR})
endif()