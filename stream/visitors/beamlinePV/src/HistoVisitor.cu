#include "SequenceVisitor.cuh"
#include "pv_beamline_histo.cuh"
#ifdef WITH_ROOT
#include "TTree.h"
#endif

template<>
void SequenceVisitor::set_arguments_size<pv_beamline_histo_t>(
  pv_beamline_histo_t::arguments_t arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers)
{
  // Set arguments size
  arguments.set_size<dev_zhisto>(host_buffers.host_number_of_selected_events[0] * (zmax - zmin) / dz);
}

template<>
void SequenceVisitor::visit<pv_beamline_histo_t>(
  pv_beamline_histo_t& state,
  const pv_beamline_histo_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  state.set_opts(dim3(host_buffers.host_number_of_selected_events[0]), 128, cuda_stream);
  state.set_arguments(
    arguments.offset<dev_atomics_velo>(),
    arguments.offset<dev_velo_track_hit_number>(),
    arguments.offset<dev_pvtracks>(),
    arguments.offset<dev_zhisto>());

  state.invoke();

  // debugging
  /*
    // Retrieve result
  cudaCheck(cudaMemcpyAsync(
    host_buffers.host_zhisto,
    arguments.offset<dev_zhisto>(),
    arguments.size<dev_zhisto>(),
    cudaMemcpyDeviceToHost,
    cuda_stream
  ));

  // Wait to receive the result
  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);

  // Check the output
  TFile * outfile = new TFile("testt.root","RECREATE");
  TTree * outtree = new TTree("PV","PV");
  int i_event = 0;
  outtree->Branch("event",&i_event);
  float z_histo;
  float z_bin;
  outtree->Branch("z_histo",&z_histo);
  outtree->Branch("z_bin",&z_bin);
  int mindex;
  outtree->Branch("index",&mindex);
  for(i_event = 0; i_event < host_buffers.host_number_of_selected_events[0]; i_event++) {
    info_cout << "number event " << i_event << std::endl;
    for (int i=0; i<Nbins; i++) {
      int index = Nbins * i_event + i;
      mindex = i;

      z_histo = host_buffers.host_zhisto[index];
      z_bin = m_zmin + i * m_dz;
      if(host_buffers.host_zhisto[index]> 5.) info_cout << "zhisto: " << i << " " << z_bin << " " <<
  host_buffers.host_zhisto[index] << std::endl << std::endl; outtree->Fill();
   }
  }
  outtree->Write();
  outfile->Close();

  */
}
