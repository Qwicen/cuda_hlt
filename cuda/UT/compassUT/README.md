# CompassUT

To configure the number of candidates to search for go to `CompassUTDefinitions.cuh` and change: 

```cpp
constexpr uint max_considered_before_found = 4;
```

A higher number considers more candidates but takes more time. It saves the best found candidate.