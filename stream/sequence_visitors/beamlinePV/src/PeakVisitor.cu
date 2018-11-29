#include "SequenceVisitor.cuh"
#include "blpv_peak.cuh"
#include "TTree.h"

template<>
void SequenceVisitor::set_arguments_size<blpv_peak_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  // Set arguments size
  arguments.set_size<dev_zpeaks>(runtime_options.number_of_events * PV::max_number_vertices);
  arguments.set_size<dev_number_of_zpeaks>(runtime_options.number_of_events);
}


template<>
void SequenceVisitor::visit<blpv_peak_t>(
  blpv_peak_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{

  state.set_opts(dim3(runtime_options.number_of_events), 1, cuda_stream);
  state.set_arguments(
    arguments.offset<dev_zhisto>(),
    arguments.offset<dev_zpeaks>(),
    arguments.offset<dev_number_of_zpeaks>()
  );


  state.invoke();

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
  outtree->Branch("z_histo",&z_histo);
  int mindex;
  outtree->Branch("index",&mindex);
  for(i_event = 0; i_event < runtime_options.number_of_events; i_event++) {
    info_cout << "number event " << i_event << std::endl;
    int Nbins = (m_zmax-m_zmin)/m_dz;
    for (int i=0; i<Nbins; i++) {
    int index = Nbins * i_event + i;
    mindex = i;
    info_cout << "zhisto: " << host_buffers.host_zhisto[index] << std::endl << std::endl;
    z_histo = host_buffers.host_zhisto[index];
    outtree->Fill();
   }
  }
  outtree->Write();
  outfile->Close();
  
  
*/


    
}
