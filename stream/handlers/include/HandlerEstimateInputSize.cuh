#pragma once

#include "HandlerDispatcher.cuh"
#include "../../sequence_setup/include/SequenceArgumentEnum.cuh"
#include <iostream>

template<typename R, typename... T>
struct HandlerEstimateInputSize : public Handler<seq::estimate_input_size, R, T...> {
  HandlerEstimateInputSize() = default;
  HandlerEstimateInputSize(R(*param_function)(T...))
  : Handler<seq::estimate_input_size, R, T...>(param_function) {}

  // Add your own methods
  void mymethod() {
    std::cout << "Wow" << std::endl;
  }
};

// Register partial specialization
template<>
struct HandlerDispatcher<seq::estimate_input_size> {
  template<typename R, typename... T>
  using H = HandlerEstimateInputSize<R, T...>;
};
