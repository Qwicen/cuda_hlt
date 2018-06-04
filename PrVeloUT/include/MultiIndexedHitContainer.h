#ifndef  MULTIINDEXEDHITCONTAINER_H
#define  MULTIINDEXEDHITCONTAINER_H 1

#include <cassert>
#include <vector>
#include <algorithm>
#include <numeric>
#include <array>

// Range V3
#include <range/v3/algorithm.hpp>
#include <range/v3/view.hpp>

namespace details
{
  template<typename C, typename Tuple, size_t ... I>
  auto emplace_tuple(C& c, Tuple&& t, std::index_sequence<I...>) -> typename C::reference
  {
    c.emplace_back ( std::get<I>(t)... );

    return c.back ();
  }
}

namespace
{

  /**
   * When C++17 is supported, all references to this function
   * should be replaced with (... * sizes ).
   *
   * Allocate one more than the actual number requested.
   *
   * Reason for this is performance: this allows us to quickly
   * return the size of the offsets (and if empty).
   ***/
  template<size_t ... S>
  constexpr size_t multiply()
  {
    size_t result = 1;

    for (auto s : { S... })
      result *= s;

    return result;
  }

  /**
   * C++11 version of multiply.
	template <typename ...Ts>
	constexpr size_t multiply()
	{
		return  1;
	}

	template <size_t I,size_t... Is>
	constexpr size_t multiply()
	{
		return   I * multiply<Is...>();
	}

   */

  /**
   * Helper function to access the value of the template
   * argument as an actual value. Only works for size_t.
   *
   * If there is a smarter way to do this, please change.
   */
  template<class T>
  size_t fv(const T& t)
  {
    return t;
  }

}
/**
 * Desirably efficient container of hits, in effect a
 * smarter wrapper around a 1D vector and an array of offsets.
 *
 * In effect, the collection of Hits* for the
 * n-dimensional detector geometry* elements is mapped to a
 * 1-dimensional vector of Hits*, together with an array
 * of offsets. Sorting by a general comparison function is
 * supported.
 *
 * Everything highlighted with the asterisk is a template argument
 * for the MultiIndexedHitContainer.
 *
 * Example use-cases are the PrFTHitHandler and the PrUTHitHandler.
 * In general, you want Hit Handlers to be the front-end to this
 * container.
 *
 * Numbering always starts at 0.
 *
 * TODO:
 *  - Write a story about why there is some special behaviour for the
 * single-hit insertion, such that an explicit sort *must* be called
 * in that situation.
 *
 *  - Main concerns: Laurent thinks it's probably beneficial to add a trivial
 * implementation of the getUniqueDetectorId for the 1D-case, which just
 * is the identity function. But be wary, as a 1D-call to the
 * getHits for a nested subdetector gives you all hits for the children
 * subdetectors, while the speed gain is really only in the 1D-Call/1D detector
 * case.
 *
 *  - Right now there is an assert for the number of arguments and the nesting of
 * subdetectors. However, there is no assert whether the sub-detector ID might
 * exceed the
 *
 *  - Support addHit([subdetectorId], hit) instead of just addHit(hit, [subdetectorId])
 *
 * @author  Roel Aaij <roel.aaij@cern.ch>,
 * 			Laurent Dufour <laurent.dufour@cern.ch>,
 * 			Renato Quagliani <Renato.Quagliani@cern.ch>,
 * 			Gerhard Raven <gerhard.raven@nikhef.nl>
 */
template<class Hit, size_t... sizes>
class MultiIndexedHitContainer
{
public:
  using offset_t = unsigned short;
  using const_reference = const Hit&;
  using value_type = Hit;

  using Hits = typename std::vector<value_type>;
  using iterator = typename Hits::iterator;
  using const_iterator = typename Hits::const_iterator;
  using const_reverse_iterator = typename Hits::const_reverse_iterator;

  /**
   * For the offsets we reserve one more, to make the calculation of the
   * number of hits quicker (can always do n(uniqueSubDetId).second - n(uniqueSubDetId).first )
   */
  using Offsets = std::array<std::pair<offset_t, offset_t>, multiply<sizes...>() >;
  using Ids = std::array<size_t, multiply<sizes...>()>;

  using HitRange=Gaudi::Range_<MultiIndexedHitContainer<Hit, sizes...> >;

  /**
   * Constructor for a relatively empty hit container.
   */
  MultiIndexedHitContainer(size_t reserve = 0)
  {
    if (reserve != 0)
      m_hits.reserve ( reserve );

    clearOffsets();
    std::fill ( begin(m_nIds), end(m_nIds), 0u);
  }

  MultiIndexedHitContainer(std::vector<Hit>&& hits, Offsets&& offsets)
  : m_hits(std::move(hits)), m_offsets(std::move(offsets))
  {

  }

  template<typename ... Args>
  std::pair<iterator, iterator> range_( Args&&...args )
  {
    auto rng = getOffset(std::forward<Args>(args)...);

    return
        {	std::next( std::begin(m_hits), rng.first ),
          std::next( std::begin(m_hits), rng.second )};
  }

  template<typename ... Args>
  HitRange range ( Args&&...args ) const
  {
    auto rng = getOffset(std::forward<Args>(args)...);

    return
        {	std::next( begin(m_hits), rng.first ),
          std::next( begin(m_hits), rng.second )};
  }

  /**
   * Returns a range of all the hits in this container.
   *
   * The sorting of this range depends on the possibly
   * used sort() method, but is always segmented
   * in unique detector ids.
   */
  HitRange range() const
  {
    return {std::begin(m_hits), std::end(m_hits)};
  }

  Hit& hit(size_t index)
  {
    return m_hits[index];
  }

  const Hit& hit(size_t index) const
  {
    return m_hits[index];
  }

  /**
   * Returns the total number of hits in this container.
   */
  size_t size() const
  {
    return m_hits.size ();
  }

  /**
   * Returns the number of hits in this container for a given
   * sub-detector.
   */
  template<typename ... Args>
  size_t size(Args&&...args) const
  {
    offset_t obegin{0}, oend{0};
    std::tie(obegin, oend) = getOffsets(std::forward(args)...);

    assert(oend >= obegin && "ill-formed offsets");
    return oend - obegin;
  }

  template<typename ... Args>
  bool empty(Args&&...args) const
  {
    offset_t obegin{0}, oend{0};
    std::tie(obegin, oend) = getOffsets(std::forward(args)...);

    assert(oend >= obegin && "ill-formed offsets");
    return oend == obegin;
  }

  inline size_t nSubDetectors() const
  {
    return multiply<sizes...>();
  }

  /**
   * Function to insert a range of hits in (detectorElementId).
   *
   * The argument are two r-valued references, but in practice this will
   * take its form as two iterators over the hits to be inserted.
   *
   * The responsibility of passing these iterators and generating the
   * hits to be inserted is found in the HitManagers
   * (e.g. {@see TStationHitManager::i_prepareHits}).
   *
   * Current practice: give a range instead of the raw begin() and end().
   *
   */
  template<typename I, typename ... Args>
  void insert(I&& b, I&& e, Args&& ... args)
  {
    auto n = std::distance ( b, e );
    m_hits.reserve ( m_hits.size () + n );

    auto subDetectorId = getUniqueDetectorElementId( std::forward<Args>(args)... );

    m_hits.insert ( std::next ( begin ( m_hits ), m_offsets[subDetectorId].first ), std::forward<I> ( b ), std::forward<I> ( e ) );
    // adjust everything 'behind' the insertion point...
    auto add_n = [n](typename Offsets::value_type& offset) {
      offset.first += n;
      offset.second += n;
    };

    // subsequent stations, move n to the right.
    m_offsets[subDetectorId].second += n;
    std::for_each ( std::next ( m_offsets, subDetectorId + 1 ), std::end ( m_offsets ), add_n );
  }

  /*
   * This implementation can be used when we have std::invoke available.
   *
		template<typename Tuple, typename ... LocArgs>
		Hit& addHit(Tuple&& cargs, LocArgs&&... subDetectorElement)
		{
			static_assert((sizeof...(subDetectorElement) <= sizeof...(sizes)),
					"The number of indices provided is strictly higher than the nesting for this subdetector.");

			auto id = getUniqueDetectorElementId ( std::forward<LocArgs>(subDetectorElement)... );
			m_ids.emplace_back ( id );
			++m_nIds[id];
			//constexpr auto nArguments = sizeof...(CtorArgs);

			return std::invoke ( [this]( auto&&... args ) -> decltype(auto)
			{	m_hits.emplace_back( std::forward<decltype(args)>(args)... );
				return m_hits.back();}, std::forward<Tuple> ( cargs ) );
		}
   */

  /**
   * Function to create a single hit in (detectorElementId).
   * A reference to the hit created is returned.
   *
   * The user of this function is responsible for giving the right
   * constructor arguments. Unexpected errors can occur when this does not
   * match the constructor of Hit.
   *
   * Note that due to the implementation of the addHit algorithm,
   * one should explicitly call setOffsets before further using the
   * hit manager.
   */
  template<typename ... CtorArgs, typename ... LocArgs>
  Hit& addHit(std::tuple<CtorArgs...>&& cargs, LocArgs&&... subDetectorElement)
  {
    auto id = getUniqueDetectorElementId ( std::forward<LocArgs>(subDetectorElement)... );
    m_ids.emplace_back ( id );
    m_nIds[id] += 1;

    return details::emplace_tuple ( m_hits, std::forward<std::tuple<CtorArgs...> > ( cargs ),
        std::index_sequence_for < CtorArgs... > {} );
  }

  template<typename COMPARE>
  void sort(const COMPARE& compare) {
    // Zip the ids and hits together to sort them together
    auto zipped = ranges::view::zip(m_ids, m_hits);

    // Sorting predicate that defers hit base sorting to the compare
    // function passed in
    auto pred = [&compare](const std::tuple<unsigned, Hit>& t1,
        const std::tuple<unsigned, Hit>& t2) {
      if (std::get<0>(t1) < std::get<0>(t2)) {
        return true;
      } else if (std::get<0>(t2) < std::get<0>(t1)) {
        return false;
      } else {
        return compare(std::get<1>(t1), std::get<1>(t2));
      }
    };

    // Sort the hits
    ranges::sort(zipped, pred);
  }

  void setOffsets(){
    // Set the offsets
    const auto nSub = nSubDetectors();
    m_offsets[0] = { 0, m_nIds[0] };
    for (size_t i = 1; i < nSub; ++i) {
      m_offsets[i] = { m_offsets[i-1].second, m_offsets[i-1].second + m_nIds[i] };
    }
  }

  template<typename COMPARE>
  bool is_sorted(const COMPARE& compare) {
    // Zip the ids and hits together
    auto zipped = ranges::view::zip(m_ids, m_hits);

    // Sorting predicate that defers hit base sorting to the compare
    // function passed in
    auto pred = [&compare](const std::tuple<unsigned, Hit>& t1,
        const std::tuple<unsigned, Hit>& t2) {
      if (std::get<0>(t1) < std::get<0>(t2)) {
        return true;
      } else if (std::get<0>(t2) < std::get<0>(t1)) {
        return false;
      } else {
        return compare(std::get<1>(t1), std::get<1>(t2));
      }
    };

    return std::is_sorted(zipped.begin(), zipped.end(), pred );
  }

  auto offsets() const
  {
    return m_offsets;
  }
private:
  std::vector<Hit> m_hits;

  Offsets m_offsets;

  /**
   * These two arrays are here to enable faster functionality
   * for adding hits one-by-one.
   */
  std::vector<unsigned> m_ids;
  Ids m_nIds;

  /**
   * Is it possible to do this in a more efficient manner?
   * The table for [ (x,y,z) -> unsigned int ] is constant
   * and can in principle already be generated at this objects
   * construction.
   *
   * Is it therefore faster to cache the outcome of this?
   */
  template<typename ... Args>
  inline size_t getUniqueDetectorElementId(Args&& ... args) const
  {
    constexpr auto nArguments = sizeof...(args);
    constexpr size_t detector_geometry[] = { sizes... };
    const size_t x_array[] = { fv(args)... }; // convert the arguments to size_t array

    static_assert((nArguments <= sizeof...(sizes)),
        "The number of indices provided is strictly higher than the nesting for this subdetector.");

    size_t uniqueDetectorElementId = 0;

    // The actual logic:
    //
    // For the geometry [1,1]
    // (0,0) -> 0
    // (0,1) -> 1
    // (1,0) -> 2
    // (1) -> 2
    // (0) -> 0
    for (unsigned int argumentId = 0; argumentId < sizeof...(sizes); argumentId++)
    {
      auto x = (sizeof...(args) - 1 < argumentId ? 0 : x_array[ argumentId ]);

      for (unsigned int detectorId = argumentId+1;
          detectorId < sizeof...(sizes);
          detectorId++)
      {
        uniqueDetectorElementId += x * detector_geometry[ detectorId ];
      }
    }

    uniqueDetectorElementId += x_array[ sizeof...(args) - 1 ];

    return uniqueDetectorElementId;
  }

  template<typename ... Args>
  inline
  typename Offsets::value_type getOffset(Args&& ... args) const
  {
    const auto uniqueDetectorElementId = getUniqueDetectorElementId( std::forward<Args>(args)... );

    return m_offsets[ uniqueDetectorElementId ];
  }

  inline void clearOffsets()
  {
    typename Offsets::value_type zero{0, 0};
    std::fill ( std::begin ( m_offsets ), std::end ( m_offsets ), zero );
  }
};
#endif
