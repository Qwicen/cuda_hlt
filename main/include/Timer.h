#pragma once

/**
 *      Timer
 *
 *      author  -   Daniel Campora
 *      email   -   dcampora@cern.ch
 *
 *      December, 2013
 *      CERN
 */

#include <ctime>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <functional>
#include <iomanip>
#include <cmath>
#include <map>

class Timer {
public:
  bool _append;
  long long unsigned timeDiff;
  struct timespec tstart, tend;

  Timer ();

  inline void start ();
  inline void stop ();
  inline void flush ();

  /**
   * Gets the elapsed time since tstart
   * @return
   */
  inline double getElapsedTime ();
  inline long long unsigned getElapsedTimeLong ();

  /**
   * Gets the accumulated time in timeDiff
   * @return
   */
  inline double get () const;
  inline long long unsigned getLong () const;

  /**
   * @brief      Gets start and end times.
   */
  inline double getStartTime () const;
  inline double getEndTime () const;

  /**
   * @brief      Gets the current time.
   */
  inline static double getCurrentTime ();
};

class TimePrinter {
private:
  double baselineTime;
  unsigned int bottomPercentile, topPercentile;
  bool outlierRemoval;

public:
  TimePrinter (
    bool outlierRemoval = false,
    unsigned int bottomPercentile = 3,
    unsigned int topPercentile = 97)
    : baselineTime(0), bottomPercentile(bottomPercentile),
      topPercentile(topPercentile), outlierRemoval(outlierRemoval) {}

  template<class V>
  std::map<std::string, double> printTimer (V& timers, const std::string& algName, bool printout=true, bool taketime=false)
  {
    std::vector<double> counter, counter_subset;
    std::for_each(timers.begin(), timers.end(), [&] (Timer& timers) { counter.push_back(timers.get()); });

    if (outlierRemoval) {
      // Counter is copied to print "as is" afterwards, and avoid confusion by nth_element reordering
      std::vector<double> counter_copy (counter);

      // Let's exclude the highest 3-percentile (anything from 97 on) and lowest 3-percentile
      const double bpt = bottomPercentile / 100.0;
      const double tpt = topPercentile / 100.0;
      nth_element(counter_copy.begin(), counter_copy.begin() + ((int) counter_copy.size() * bpt), counter_copy.end());
      nth_element(counter_copy.begin(), counter_copy.begin() + ((int) counter_copy.size() * tpt), counter_copy.end());
      counter_subset = std::vector<double>(counter_copy.begin() + ((int) counter_copy.size() * bpt),
                                           counter_copy.begin() + ((int) counter_copy.size() * tpt));
    }
    else {
      counter_subset = counter;
    }

    const double max = * std::max_element(counter_subset.begin(), counter_subset.end());
    const double min = * std::min_element(counter_subset.begin(), counter_subset.end());
    const double sum = std::accumulate(counter_subset.begin(), counter_subset.end(), 0.0);
    const double mean = sum / counter_subset.size();
    const double sq_sum = std::inner_product(counter_subset.begin(), counter_subset.end(), counter_subset.begin(), 0.0);
    const double stdev = std::sqrt(sq_sum / counter_subset.size() - mean * mean);

    double speedup = 1.0;
    if (taketime) baselineTime = sum;
    else          speedup = baselineTime / sum;

    // Write out all timing numbers, in case of posterior analysis
    std::cout << algName << " timers:" << std::endl;
    if (outlierRemoval) std::cout << " bottom percentile: " << bottomPercentile << ", top percentile: " << topPercentile << std::endl;
    std::cout << " mean: " << mean << " sum: " << sum << " min: " << min << " max: " << max << " stddev: "
      << stdev << (baselineTime!=0.f ? " speedup: " + std::to_string(speedup) + "x" : "") << std::endl;

    if (printout) {
      std::cout << std::endl << " raw timers: " << std::setw(2) << std::setprecision(2);
      for (const auto& c : counter) std::cout << c << ", ";
      std::cout << std::setw(11) << std::setprecision(6) << std::endl << std::endl;
    }

    return {{"mean", mean}, {"sum", sum}, {"stdev", stdev}, {"speedup", speedup}, {"min", min}, {"max", max}};
  }

  template<class V>
  std::map<std::string, double> printWeightedTimer (
    const V& weightedTimers,
    const std::string& algName,
    bool printout=true,
    bool taketime=false
  ) {
    // Remove the zero weighted timers
    std::vector<std::pair<int, Timer>> nonzeroWeightedTimers;
    for (const auto& wt : weightedTimers) nonzeroWeightedTimers.push_back(wt);
    auto itend = std::remove_if(nonzeroWeightedTimers.begin(), nonzeroWeightedTimers.end(), [] (const std::pair<int, Timer>& wt) {
      return wt.first == 0;
    });
    nonzeroWeightedTimers.erase(itend, nonzeroWeightedTimers.end());

    std::vector<double> counter, counter_subset;
    std::for_each(nonzeroWeightedTimers.begin(), nonzeroWeightedTimers.end(), [&] (const std::pair<int, Timer>& wt) {
      counter.push_back(wt.second.get() / wt.first);
    });

    if (outlierRemoval && counter.size() >= 3) {
      // Counter is copied to print "as is" afterwards, and avoid confusion by nth_element reordering
      std::vector<double> counter_copy (counter);

      // Let's exclude the highest 3-percentile (anything from 97 on) and lowest 3-percentile
      const double bpt = bottomPercentile / 100.0;
      const double tpt = topPercentile / 100.0;
      nth_element(counter_copy.begin(), counter_copy.begin() + ((int) counter_copy.size() * bpt), counter_copy.end());
      nth_element(counter_copy.begin(), counter_copy.begin() + ((int) counter_copy.size() * tpt), counter_copy.end());
      counter_subset = std::vector<double>(counter_copy.begin() + ((int) counter_copy.size() * bpt),
                                           counter_copy.begin() + ((int) counter_copy.size() * tpt));
    }
    else {
      counter_subset = counter;
    }

    if (counter_subset.size() > 0) {
      const double max = * std::max_element(counter_subset.begin(), counter_subset.end());
      const double min = * std::min_element(counter_subset.begin(), counter_subset.end());
      const double sum = std::accumulate(counter_subset.begin(), counter_subset.end(), 0.0);
      const double mean = sum / counter_subset.size();
      const double sq_sum = std::inner_product(counter_subset.begin(), counter_subset.end(), counter_subset.begin(), 0.0);
      const double stdev = std::sqrt(sq_sum / counter_subset.size() - mean * mean);
      
      double rawsum = 0.0;
      std::for_each(nonzeroWeightedTimers.begin(), nonzeroWeightedTimers.end(), [&rawsum] (const std::pair<int, Timer>& wt) {
        rawsum += wt.second.get();
      });

      double speedup = 1.0;
      if (taketime) baselineTime = sum;
      else          speedup = baselineTime / sum;

      // Write out all timing numbers, in case of posterior analysis
      std::cout << algName << " timers:" << std::endl;
      if (nonzeroWeightedTimers.size() != weightedTimers.size()) {
        std::cout << " note: There were " << weightedTimers.size() - nonzeroWeightedTimers.size()
          << " measurements with weight 0 that were left out" << std::endl;
      }
      if (outlierRemoval) std::cout << " bottom percentile: " << bottomPercentile << ", top percentile: " << topPercentile << std::endl;
      std::cout << " mean: " << mean << " sum: " << sum << " min: " << min << " max: " << max << " stddev: "
        << stdev << (baselineTime!=0.f ? " speedup: " + std::to_string(speedup) + "x" : "")
        << " rawsum: " << rawsum << std::endl;
      
      if (printout) {
        std::cout << " raw weights: " << std::setw(2) << std::setprecision(2);
        for (const auto& wt : nonzeroWeightedTimers) std::cout << wt.first << ", ";
        std::cout << std::endl << " raw timers: ";
        for (const auto& wt : nonzeroWeightedTimers) std::cout << wt.second.get() << ", ";
        std::cout << std::setw(11) << std::setprecision(6) << std::endl << std::endl;
      }

      return {{"mean", mean}, {"sum", sum}, {"stdev", stdev}, {"speedup", speedup}, {"min", min}, {"max", max}, {"rawsum", rawsum}};
    } else {
      return {{"mean", 0}, {"sum", 0}, {"stdev", 0}, {"speedup", 0}, {"min", 0}, {"max", 0}};
    }
  }
};

#include "Timer.h_impl"
