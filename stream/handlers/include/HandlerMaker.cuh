#pragma once

#include "HandlerDispatcher.cuh"

// Note: Add here additional custom handlers
#include "HandlerEstimateInputSize.cuh"

template<unsigned long I>
struct HandlerMaker {
  template<typename R, typename... T>
  static typename HandlerDispatcher<I>::H<R, T...> make_handler(R(f)(T...)) {
    return typename HandlerDispatcher<I>::H<R, T...>{f};
  }
};
