#include "Test_Selector.hpp"
#include <iostream>
#include "boost/program_options.hpp" 
#include <iterator>
#include "args.h"
#include <ctime>
#include "TFile.h"
#include "TTree.h"


using namespace std;
#include "globbing.h"

int main(int ac, const char** av){
  //add your favourite flags to parse for a given algorithm for example [search windows]
  args::ArgumentParser parser("Parser for AOS Test tracking algorithm");
  args::ValueFlag<int> nevents(parser, "nevents", "N events to run over", {'n'});
  args::HelpFlag help(parser, "help", "Diplay this help menu", {'h',"help"});
  try{ 
    parser.ParseCLI(ac, av); 
  }
  catch (args::Help){
    std::cout << parser;
    return 0;  
  }
  catch (args::ParseError e)
  { 
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1; 
  }
  catch (args::ValidationError e)
  {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;  
  }

  
  int nbevents = args::get(nevents);
  using namespace std;   
  // using namespace SOA;

  //--- Load the tuple
  
  TString path_data = TString( getenv("DATAPATH"))+TString("/*.root");
  std::cout<<"path_data = "<< path_data<<std::endl;
  std::vector<TString> InputFiles = glob(path_data.Data() );

  //just check if the argument passed is enough given the files to load (i.e. the nb events)
  if( InputFiles.size() < nbevents || nbevents == 0){
    nbevents = InputFiles.size();
  }
  int evtId = 0;
  std::cout<<"Will process N events = "<< nbevents <<std::endl;
  for( int evtId = 0; evtId < nbevents ; ++ evtId ){
    TFile file(InputFiles[evtId],"READ");
    TTree *tree = (TTree*)file.Get("Hits_detectors");    
    std::cout<<"Add input file"<< InputFiles[evtId] << std::endl;
    assert( tree);
    Test_Selector * looper = new Test_Selector();
    std::cout<<"Processing event "<<evtId<<" / "<< nbevents <<std::endl;
    std::cout<< "Init" << std::endl;
    looper->Init( tree);
    std::cout<< "Begin" << std::endl;
    looper->Begin( tree);
    std::cout<< "Process" << std::endl;
    looper->GetUTVeloTree(evtId);


    //the idea is to have a kind of
    //MyFavourite VPContainer = looper->GetVeloHits();
    
    //write your own velo tracking: class VeloTracking
    //VeloTracking vptracking( VPContainer);
    //vptracking->SetConfigurations.....
    //vptracking->dotracking();
    //---- vector of lhcbids of the hits on track
    //std::vector< std::vector<int> >  tracks = vptracking->getTracks();
    //--- something like this
    //std::map< lhcbid, vector< MCParticle> > looper->getMapAssociation();

    //-- outside this loop
    //-- MyChecker Checker();
    //inside this loop Checker.AddEvent( tracks, map ) ;
    //after the loop, Checker.PrintResults();
    //---also make histograms eventually as in the PrChecker....!
    delete looper;
  }
 }
