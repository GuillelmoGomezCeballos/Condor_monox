#include <iostream>
#include <string>
#include "TFile.h"
#include "TTree.h"

void merge(std::string iFile0="ttbar.root",std::string iFile1="ttbar_mva_698.root",std::string iName0="bdt_monox_",std::string iName1="jetC2b1,jetQJetVol,jetMassPruned") { 
  TFile *lFile0 = new TFile(iFile0.c_str(),"UPDATE");
  TTree *lTree0 = (TTree*) lFile0->FindObjectAny("DMSTree");

  TFile *lFile1 = new TFile(iFile1.c_str());
  TTree *lTree1 = (TTree*) lFile1->FindObjectAny("exampleEventsNtuple");
  
  std::string iName = iName0 + iName1;
  
  float lBDT0 = 0; lTree1->SetBranchAddress(iName.c_str(),&lBDT0); 
  TBranch* lBranch = lTree1->GetBranch(iName.c_str());
  
  float lBDT  = 0; 
  TBranch* lBDTBranch = lTree0->Branch(iName.c_str(),&lBDT,(iName+"/F").c_str());
  int lNEntries = lTree0->GetEntriesFast();

  for(int i0 = 0; i0 < lNEntries; i0++) {
    lBranch->GetEntry(i0); 
    lBDT = lBDT0;
    //lTree0->GetEntry(i0); 
    lBDTBranch->Fill();
  }
  lFile0->cd();
  lTree0->Write();
}
