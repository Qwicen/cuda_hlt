#pragma once

/**
 *      Timer
 *
 *      author  -   Daniel Campora
 *      email   -   dcampora@cern.ch
 *
 *      March, 2018
 *      CERN
 */

#include <chrono>

class Timer {
private:
  // The timers are in seconds, stored in double (ratio 1:1 by default)
  std::chrono::duration<double> accumulated_elapsed_time;
  std::chrono::high_resolution_clock::time_point start_time;
  std::chrono::high_resolution_clock::time_point stop_time;

public:
  Timer();

  /**
   * @brief Starts the timer
   */
  void start();

  /**
   * @brief Stops the timer
   */
  void stop();

  /**
   * @brief Flushes the timer
   */
  void flush();

  /**
   * @brief Flushes the timer and starts it
   */
  void restart();

  /**
   * @brief Gets the elapsed time since start
   */
  double get_elapsed_time() const;

  /**
   * @brief Gets the accumulated time
   */
  double get() const;

  /**
   * @brief Gets start time
   */
  double get_start_time() const;

  /**
   * @brief Gets stop time
   */
  double get_stop_time() const;

  /**
   * @brief Gets the current time
   */
  static double get_current_time();
};
