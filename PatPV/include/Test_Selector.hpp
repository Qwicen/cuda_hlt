//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Sat May 19 07:47:57 2018 by ROOT version 6.10/08
// from TTree Hits_detectors/Hits_detectors
// found on file: DumperFTUTHits_runNb_6719549_evtNb_21912.root
//////////////////////////////////////////////////////////

#ifndef Test_Selector_h
#define Test_Selector_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TSelector.h>

#include "global.h"
#include  "PVSeedTool.h"
#include  "AdaptivePV3DFitter.h"

// Header file for the classes stored in the TTree if any.
#include <vector>
#include <iostream>
using namespace std;












class Test_Selector : public TSelector {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain

  // Fixed size dimensions of array or collections stored in the TTree if any.






   // Declaration of leaf types
   Int_t           nVeloHits;
   vector<float>   *Velo_x;
   vector<float>   *Velo_y;
   vector<float>   *Velo_z;
   vector<int>     *Velo_Module;
   vector<int>     *Velo_Sensor;
   vector<int>     *Velo_Station;
   vector<unsigned int> *Velo_lhcbID;
   Int_t           nFTHits;
   vector<float>   *FT_x;
   vector<float>   *FT_z;
   vector<float>   *FT_w;
   vector<float>   *FT_dxdy;
   vector<float>   *FT_YMin;
   vector<float>   *FT_YMax;
   vector<int>     *FT_hitPlaneCode;
   vector<int>     *FT_hitzone;
   vector<unsigned int> *FT_lhcbID;
   Int_t           nUTHits;
   vector<float>   *UT_cos;
   vector<float>   *UT_cosT;
   vector<float>   *UT_dxDy;
   vector<bool>    *UT_highthreshold;
   vector<unsigned int> *UT_lhcbID;
   vector<int>     *UT_planeCode;
   vector<float>   *UT_sinT;
   vector<int>     *UT_size;
   vector<float>   *UT_tanT;
   vector<float>   *UT_weight;
   vector<float>   *UT_xAtYEq0;
   vector<float>   *UT_xAtYMid;
   vector<float>   *UT_xMax;
   vector<float>   *UT_xMin;
   vector<float>   *UT_xT;
   vector<float>   *UT_yBegin;
   vector<float>   *UT_yEnd;
   vector<float>   *UT_yMax;
   vector<float>   *UT_yMid;
   vector<float>   *UT_yMin;
   vector<float>   *UT_zAtYEq0;
   Bool_t          fullInfo;
   Bool_t          hasSciFi;
   Bool_t          hasUT;
   Bool_t          hasVelo;
   Bool_t          isDown;
   Bool_t          isDown_noVelo;
   Bool_t          isLong;
   Int_t           key;
   Bool_t          isLong_andUT;
   Double_t        p;
   Double_t        pt;
   Int_t           pid;
   Double_t        eta;
   Double_t        ovtx_x;
   Double_t        ovtx_y;
   Double_t        ovtx_z;
   Bool_t          fromBeautyDecay;
   Bool_t          fromCharmDecay;
   Bool_t          fromStrangeDecay;
   Int_t           DecayOriginMother_pid;

   // List of branches
   TBranch        *b_nVeloHits;   //!
   TBranch        *b_Velo_x;   //!
   TBranch        *b_Velo_y;   //!
   TBranch        *b_Velo_z;   //!
   TBranch        *b_Velo_Module;   //!
   TBranch        *b_Velo_Sensor;   //!
   TBranch        *b_Velo_Station;   //!
   TBranch        *b_Velo_lhcbID;   //!
   TBranch        *b_nFTHits;   //!
   TBranch        *b_FT_x;   //!
   TBranch        *b_FT_z;   //!
   TBranch        *b_FT_w;   //!
   TBranch        *b_FT_dxdy;   //!
   TBranch        *b_FT_YMin;   //!
   TBranch        *b_FT_YMax;   //!
   TBranch        *b_FT_hitPlaneCode;   //!
   TBranch        *b_FT_hitzone;   //!
   TBranch        *b_FT_lhcbID;   //!
   TBranch        *b_nUTHits;   //!
   TBranch        *b_UT_cos;   //!
   TBranch        *b_UT_cosT;   //!
   TBranch        *b_UT_dxDy;   //!
   TBranch        *b_UT_highthreshold;   //!
   TBranch        *b_UT_lhcbID;   //!
   TBranch        *b_UT_planeCode;   //!
   TBranch        *b_UT_sinT;   //!
   TBranch        *b_UT_size;   //!
   TBranch        *b_UT_tanT;   //!
   TBranch        *b_UT_weight;   //!
   TBranch        *b_UT_xAtYEq0;   //!
   TBranch        *b_UT_xAtYMid;   //!
   TBranch        *b_UT_xMax;   //!
   TBranch        *b_UT_xMin;   //!
   TBranch        *b_UT_xT;   //!
   TBranch        *b_UT_yBegin;   //!
   TBranch        *b_UT_yEnd;   //!
   TBranch        *b_UT_yMax;   //!
   TBranch        *b_UT_yMid;   //!
   TBranch        *b_UT_yMin;   //!
   TBranch        *b_UT_zAtYEq0;   //!
   TBranch        *b_fullInfo;   //!
   TBranch        *b_hasSciFi;   //!
   TBranch        *b_hasUT;   //!
   TBranch        *b_hasVelo;   //!
   TBranch        *b_isDown;   //!
   TBranch        *b_isDown_noVelo;   //!
   TBranch        *b_isLong;   //!
   TBranch        *b_key;   //!
   TBranch        *b_isLong_andUT;   //!
   TBranch        *b_p;   //!
   TBranch        *b_pt;   //!
   TBranch        *b_pid;   //!
   TBranch        *b_eta;   //!
   TBranch        *b_ovtx_x;   //!
   TBranch        *b_ovtx_y;   //!
   TBranch        *b_ovtx_z;   //!
   TBranch        *b_fromBeautyDecay;   //!
   TBranch        *b_fromCharmDecay;   //!
   TBranch        *b_fromStrangeDecay;   //!
   TBranch        *b_DecayOriginMother_pid;   //!

   Test_Selector(TTree * /*tree*/ =0) : fChain(0) { }
   virtual ~Test_Selector() { }
   virtual Int_t   Version() const { return 2; }
   virtual void    Begin(TTree *tree);
   virtual void    SlaveBegin(TTree *tree);
   virtual void    Init(TTree *tree);
   virtual Bool_t  Notify();
  virtual Bool_t Process( Long64_t entry);
   Bool_t  Process();
   virtual Int_t   GetEntry(Long64_t entry, Int_t getall = 0) { return fChain ? fChain->GetTree()->GetEntry(entry, getall) : 0; }
   virtual void    SetOption(const char *option) { fOption = option; }
   virtual void    SetObject(TObject *obj) { fObject = obj; }
   virtual void    SetInputList(TList *input) { fInput = input; }
   virtual TList  *GetOutputList() const { return fOutput; }
   virtual void    SlaveTerminate();
   virtual void    Terminate();
   void GetVeloTracks(int evtId);

   // ClassDef(Test_Selector,0);  // <-------- THe MakeSelector make this line, comment it out when creating the class
};

#endif

#ifdef Test_Selector_cxx
void Test_Selector::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   
   // Set object pointer
   Velo_x = 0;
   Velo_y = 0;
   Velo_z = 0;
   Velo_Module = 0;
   Velo_Sensor = 0;
   Velo_Station = 0;
   Velo_lhcbID = 0;
   FT_x = 0;
   FT_z = 0;
   FT_w = 0;
   FT_dxdy = 0;
   FT_YMin = 0;
   FT_YMax = 0;
   FT_hitPlaneCode = 0;
   FT_hitzone = 0;
   FT_lhcbID = 0;
   UT_cos = 0;
   UT_cosT = 0;
   UT_dxDy = 0;
   UT_highthreshold = 0;
   UT_lhcbID = 0;
   UT_planeCode = 0;
   UT_sinT = 0;
   UT_size = 0;
   UT_tanT = 0;
   UT_weight = 0;
   UT_xAtYEq0 = 0;
   UT_xAtYMid = 0;
   UT_xMax = 0;
   UT_xMin = 0;
   UT_xT = 0;
   UT_yBegin = 0;
   UT_yEnd = 0;
   UT_yMax = 0;
   UT_yMid = 0;
   UT_yMin = 0;
   UT_zAtYEq0 = 0;
   // Set branch addresses and branch pointers
   if (!tree){
     cout<< "NULL POINTER TTREE, FIX NAMES"<<endl;  
     return;
   }
   fChain = tree;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("nVeloHits", &nVeloHits, &b_nVeloHits);
   fChain->SetBranchAddress("Velo_x", &Velo_x, &b_Velo_x);
   fChain->SetBranchAddress("Velo_y", &Velo_y, &b_Velo_y);
   fChain->SetBranchAddress("Velo_z", &Velo_z, &b_Velo_z);
   fChain->SetBranchAddress("Velo_Module", &Velo_Module, &b_Velo_Module);
   fChain->SetBranchAddress("Velo_Sensor", &Velo_Sensor, &b_Velo_Sensor);
   fChain->SetBranchAddress("Velo_Station", &Velo_Station, &b_Velo_Station);
   fChain->SetBranchAddress("Velo_lhcbID", &Velo_lhcbID, &b_Velo_lhcbID);
   fChain->SetBranchAddress("nFTHits", &nFTHits, &b_nFTHits);
   fChain->SetBranchAddress("FT_x", &FT_x, &b_FT_x);
   fChain->SetBranchAddress("FT_z", &FT_z, &b_FT_z);
   fChain->SetBranchAddress("FT_w", &FT_w, &b_FT_w);
   fChain->SetBranchAddress("FT_dxdy", &FT_dxdy, &b_FT_dxdy);
   fChain->SetBranchAddress("FT_YMin", &FT_YMin, &b_FT_YMin);
   fChain->SetBranchAddress("FT_YMax", &FT_YMax, &b_FT_YMax);
   fChain->SetBranchAddress("FT_hitPlaneCode", &FT_hitPlaneCode, &b_FT_hitPlaneCode);
   fChain->SetBranchAddress("FT_hitzone", &FT_hitzone, &b_FT_hitzone);
   fChain->SetBranchAddress("FT_lhcbID", &FT_lhcbID, &b_FT_lhcbID);
   fChain->SetBranchAddress("nUTHits", &nUTHits, &b_nUTHits);
   fChain->SetBranchAddress("UT_cos", &UT_cos, &b_UT_cos);
   fChain->SetBranchAddress("UT_cosT", &UT_cosT, &b_UT_cosT);
   fChain->SetBranchAddress("UT_dxDy", &UT_dxDy, &b_UT_dxDy);
   fChain->SetBranchAddress("UT_highthreshold", &UT_highthreshold, &b_UT_highthreshold);
   fChain->SetBranchAddress("UT_lhcbID", &UT_lhcbID, &b_UT_lhcbID);
   fChain->SetBranchAddress("UT_planeCode", &UT_planeCode, &b_UT_planeCode);
   fChain->SetBranchAddress("UT_sinT", &UT_sinT, &b_UT_sinT);
   fChain->SetBranchAddress("UT_size", &UT_size, &b_UT_size);
   fChain->SetBranchAddress("UT_tanT", &UT_tanT, &b_UT_tanT);
   fChain->SetBranchAddress("UT_weight", &UT_weight, &b_UT_weight);
   fChain->SetBranchAddress("UT_xAtYEq0", &UT_xAtYEq0, &b_UT_xAtYEq0);
   fChain->SetBranchAddress("UT_xAtYMid", &UT_xAtYMid, &b_UT_xAtYMid);
   fChain->SetBranchAddress("UT_xMax", &UT_xMax, &b_UT_xMax);
   fChain->SetBranchAddress("UT_xMin", &UT_xMin, &b_UT_xMin);
   fChain->SetBranchAddress("UT_xT", &UT_xT, &b_UT_xT);
   fChain->SetBranchAddress("UT_yBegin", &UT_yBegin, &b_UT_yBegin);
   fChain->SetBranchAddress("UT_yEnd", &UT_yEnd, &b_UT_yEnd);
   fChain->SetBranchAddress("UT_yMax", &UT_yMax, &b_UT_yMax);
   fChain->SetBranchAddress("UT_yMid", &UT_yMid, &b_UT_yMid);
   fChain->SetBranchAddress("UT_yMin", &UT_yMin, &b_UT_yMin);
   fChain->SetBranchAddress("UT_zAtYEq0", &UT_zAtYEq0, &b_UT_zAtYEq0);
   fChain->SetBranchAddress("fullInfo", &fullInfo, &b_fullInfo);
   fChain->SetBranchAddress("hasSciFi", &hasSciFi, &b_hasSciFi);
   fChain->SetBranchAddress("hasUT", &hasUT, &b_hasUT);
   fChain->SetBranchAddress("hasVelo", &hasVelo, &b_hasVelo);
   fChain->SetBranchAddress("isDown", &isDown, &b_isDown);
   fChain->SetBranchAddress("isDown_noVelo", &isDown_noVelo, &b_isDown_noVelo);
   fChain->SetBranchAddress("isLong", &isLong, &b_isLong);
   fChain->SetBranchAddress("key", &key, &b_key);
   fChain->SetBranchAddress("isLong_andUT", &isLong_andUT, &b_isLong_andUT);
   fChain->SetBranchAddress("p", &p, &b_p);
   fChain->SetBranchAddress("pt", &pt, &b_pt);
   fChain->SetBranchAddress("pid", &pid, &b_pid);
   fChain->SetBranchAddress("eta", &eta, &b_eta);
   fChain->SetBranchAddress("ovtx_x", &ovtx_x, &b_ovtx_x);
   fChain->SetBranchAddress("ovtx_y", &ovtx_y, &b_ovtx_y);
   fChain->SetBranchAddress("ovtx_z", &ovtx_z, &b_ovtx_z);
   fChain->SetBranchAddress("fromBeautyDecay", &fromBeautyDecay, &b_fromBeautyDecay);
   fChain->SetBranchAddress("fromCharmDecay", &fromCharmDecay, &b_fromCharmDecay);
   fChain->SetBranchAddress("fromStrangeDecay", &fromStrangeDecay, &b_fromStrangeDecay);
   fChain->SetBranchAddress("DecayOriginMother_pid", &DecayOriginMother_pid, &b_DecayOriginMother_pid);
  
}

Bool_t Test_Selector::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}








#endif // #ifdef Test_Selector_cxx
