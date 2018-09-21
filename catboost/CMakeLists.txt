LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}") 
set(flatbuffers_home "${CMAKE_SOURCE_DIR}/../flatbuffers")
set(catboost_home "${CMAKE_SOURCE_DIR}/../catboost")

find_package(flatbuffers REQUIRED)
find_package(catboost REQUIRED)

include_directories(include)
include_directories(${flatbuffers_INCLUDE_DIRS})
include_directories(${catboost_INCLUDE_DIRS})

function(compile_flatbuffers_schema_to_cpp SRC_FBS)
    string(REGEX REPLACE "\\.fbs$" "_generated.h" GEN_HEADER ${SRC_FBS})
    add_custom_command(
            OUTPUT ${GEN_HEADER}
            COMMAND ${flatbuffers_FLATC} -c --gen-mutable
            -o "${CMAKE_CURRENT_SOURCE_DIR}/include"
            --reflect-names
            -I "${catboost_home}"
            "${catboost_FBS}/${SRC_FBS}")
endfunction()

compile_flatbuffers_schema_to_cpp(features.fbs)
compile_flatbuffers_schema_to_cpp(ctr_data.fbs)
compile_flatbuffers_schema_to_cpp(model.fbs)

set(SRCS
    src/Catboost.cpp
    src/TestCatboost.cpp
    features_generated.h
    ctr_data_generated.h
    model_generated.h)

add_library(Catboost ${SRCS})


