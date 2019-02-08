#include "Timer.h"

Timer::Timer() : accumulated_elapsed_time(std::chrono::duration<double>::zero())
{
  start_time = std::chrono::high_resolution_clock::now();
}

void Timer::start() { start_time = std::chrono::high_resolution_clock::now(); }

void Timer::stop()
{
  stop_time = std::chrono::high_resolution_clock::now();
  accumulated_elapsed_time += stop_time - start_time;
}

void Timer::flush() { accumulated_elapsed_time = std::chrono::duration<double>::zero(); }

void Timer::restart()
{
  flush();
  start();
}

double Timer::get_elapsed_time() const
{
  return (std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::high_resolution_clock::now() - start_time))
    .count();
}

double Timer::get() const { return accumulated_elapsed_time.count(); }

double Timer::get_start_time() const { return start_time.time_since_epoch().count(); }

double Timer::get_stop_time() const { return stop_time.time_since_epoch().count(); }

double Timer::get_current_time()
{
  return std::chrono::duration_cast<std::chrono::duration<double>>(
           std::chrono::high_resolution_clock::now().time_since_epoch())
    .count();
}
