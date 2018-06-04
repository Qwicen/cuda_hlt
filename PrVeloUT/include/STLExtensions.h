#ifndef LHCBKERNEL_STLExtensions
#define LHCBKERNEL_STLExtensions

// This header contains methods that are likely to become parts of the STL
// at some point in the future, but currently are not available.

#include <array>
#include <type_traits>
#include "GaudiKernel/invoke.h"

#ifdef NDEBUG
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#define GSL_UNENFORCED_ON_CONTRACT_VIOLATION
#endif
#include "gsl/span"
#ifdef NDEBUG
#pragma GCC diagnostic pop
#endif

namespace LHCb
{
  /// An implementation (cut down) of 'std::make_array' based on
  /// http://en.cppreference.com/w/cpp/experimental/make_array
  /// Once we target gcc 6.1 or newer only, this can be removed
  /// and <experimental/array> directly used instead.
  template <typename... T>
  constexpr decltype(auto) make_array( T && ... values )
  {
    return std::array< std::decay_t< std::common_type_t<T... > >, sizeof...(T) >
           { std::forward<T>(values)... };
  }

  // TODO: when we switch to C++17, deprecate, and point to std::invoke instead;
  using Gaudi::invoke;

  // TODO: when we use a more recent version of range-v3, switch to its version of span
  using gsl::span;

  // TODO: remove with a more recent version of GSL:
  //
  template <class ElementType>
  auto make_span(ElementType* ptr, typename span<ElementType>::index_type count)
  {
      return span<ElementType>(ptr, count);
  }

  template <class ElementType>
  auto make_span(ElementType* firstElem, ElementType* lastElem)
  {
      return span<ElementType>(firstElem, lastElem);
  }

  template <class ElementType, std::size_t N>
  auto make_span(ElementType (&arr)[N])
  {
      return span<ElementType, N>(arr);
  }

  template <class ElementType, std::size_t N>
  auto make_span(std::array<ElementType,N>& arr)
  {
      return span<ElementType,N>(arr);
  }

  template <class ElementType, std::size_t N>
  auto make_span(const std::array<ElementType,N>& arr)
  {
      return span<const ElementType,N>(arr);
  }

  template <class Container>
  auto make_span(Container& cont)
  {
      return span<typename Container::value_type>(cont);
  }

  template <class Container>
  auto make_span(const Container& cont)
  {
      return span<const typename Container::value_type>(cont);
  }

  template <class Ptr>
  auto make_span(Ptr& cont, std::ptrdiff_t count)
  {
      return span<typename Ptr::element_type>(cont, count);
  }

}

// gsl::span has no `front` or `back`
// whereas range::span does -- so until we get a version of range::v3 which
// includes range::span, we provide some free standing functions to provide
// `front` and `back`...
namespace gsl {
   template <typename T, std::ptrdiff_t N>
   decltype(auto) front(span<T,N> s) { return *s.begin(); }

   template <typename T, std::ptrdiff_t N>
   decltype(auto) back (span<T,N> s) { return *s.rbegin(); }
}


#endif // LHCBKERNEL_STLExtensions
