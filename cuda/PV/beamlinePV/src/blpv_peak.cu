#include "blpv_peak.cuh"



__global__ void blpv_peak(
  float* dev_zhisto,
  float* dev_zpeaks,
  uint* dev_number_of_zpeaks) {


  const int Nbins = (m_zmax-m_zmin)/m_dz ; 

  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;


  float* zhisto = dev_zhisto + Nbins * event_number;
  float* zpeaks = dev_zpeaks + PV::max_number_vertices * event_number;
  uint number_of_peaks = 0;

  //the parameters to find peaks can and should be fiddled with
  for(uint i = 2; i < Nbins-2; i++) {
    if(zhisto[i] > zhisto[i -1] && zhisto[i] > zhisto[i+1] && (zhisto[i] + zhisto[i-1] + zhisto[i+1]+ zhisto[i-2] + zhisto[i+2] > 2.5 ) && zhisto[i] > 1.5 ) {
      zpeaks [event_number * PV::max_number_vertices + number_of_peaks] = zhisto[i];
      number_of_peaks++;
    }
  }

  dev_number_of_zpeaks[event_number] = number_of_peaks;

}