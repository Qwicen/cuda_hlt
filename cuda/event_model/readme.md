Event Model
============

The event model includes objects to store hits, track candidates (collection of 
hit indices) and tracks. The format is the same for the various sub-detectors. 

### Hits
Hit variables are stored in structures of arrays (SoAs). The allocated size corresponds
exactly to the number of htis / clusters present in the sub-detector. This structure is used
for the output of the decoding / clustering algorithm and as input for the pattern recognition algorithm.

### Track Candidates
Track Candidates are stored in a TrackHits object containing the indices to the hits
in the SoA for those hits that represent a track candidate. Additional information
necessary for the pattern recognition step might be added, such as the `qop`. 

### Consolidated Tracks
After the pattern recognition step, a consolidation of the hit SoAs is applied, such
that only those hits belonging to a track remain in memory. More specifially, for every track in
every event the hits contributing to this track are written into the hit SoA consecutively.
This means that if one hit is used in two different tracks, it will be present 
in the coalesced hit SoA twice in different locations.
The consolidation precedure saves memory and 
hits belonging to one track are coalesced. In general, the consolidated hit SoAs have the same format as
the hit SoAs described above, but their size is smaller. 
The consolidated tracks are defined in 
`common/include/ConsolidatedTypes.cuh`. Sub-detector specific implementations
inherit from these general types. 

Consolidated tracks are based on two pieces of information:

   * An array with the number of tracks in every event
   * An array with the offset to every track for every event. This offset describes
   where the hits for this track are to be found in the SoA of consolidated hits.


### States
Two different types of states exist in the current event model:

   * VeloState: Containing the position `x, y, z` and slopes `tx, ty` as well as a reduced covariance matrix with only those entries that are filled in the straight line fit / simplified Kalman filter for the Velo where the x- and y-dimensions are assumed to be independent.
   * FullState: Containing the position `x, y, z` and slopes `tx, ty` as well as all entries of the covariance matrix defined after the SciFi pattern recognition step.
   

[This](https://indico.cern.ch/event/692177/contributions/3252276/subcontributions/269262/attachments/1771794/2879440/Allen_event_model_DvB.pdf) presentation gives an overview of the event model.

