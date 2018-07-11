#pragma once

#include "Handler.cuh"

template<unsigned long I>
struct HandlerDispatcher {
  template<typename R, typename... T>
  using H = Handler<I, R, T...>;
};
