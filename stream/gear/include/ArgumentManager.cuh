#pragma once

#include <tuple>

/**
 * @brief Helper class to generate arguments based on
 *        the information provided by the base_pointer and offsets
 */
template<typename Tuple>
struct ArgumentManager {
  Tuple arguments_tuple;
  char* base_pointer;

  ArgumentManager() = default;

  void set_base_pointer(char* param_base_pointer) {
    base_pointer = param_base_pointer;
  }

  template<typename T>
  auto offset() const {
    auto pointer = std::get<T>(arguments_tuple).offset;
    return reinterpret_cast<typename T::type*>(pointer);
  }

  template<typename T>
  size_t size() const {
    return std::get<T>(arguments_tuple).size;
  }

  template<typename T>
  void set_offset(uint offset) {
    std::get<T>(arguments_tuple).offset = base_pointer + offset;
  }

  template<typename T>
  void set_size(size_t size) {
    std::get<T>(arguments_tuple).size = size * sizeof(typename T::type);
  }
};

/**
 * @brief Manager of argument references for every handler.
 */
template<typename Arguments>
struct ArgumentRefManager;

template<typename... Arguments>
struct ArgumentRefManager<std::tuple<Arguments...>> {
  using TupleToReferences = std::tuple<Arguments&...>;
  TupleToReferences m_arguments;

  ArgumentRefManager(TupleToReferences arguments) : m_arguments(arguments) {}

  template<typename T>
  auto offset() const {
    auto pointer = std::get<T&>(m_arguments).offset;
    return reinterpret_cast<typename T::type*>(pointer);
  }

  template<typename T>
  size_t size() const {
    return std::get<T&>(m_arguments).size;
  }

  template<typename T>
  void set_size(size_t size) {
    std::get<T&>(m_arguments).size = size * sizeof(typename T::type);
  }
};