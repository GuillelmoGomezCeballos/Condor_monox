#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TStopwatch.h"
#include "TChain.h"
#include "TH1D.h"

#include "MitMonoJet/Core/MitDMSTree.h"
#include "MitMonoJet/macros/functions.C"

using namespace std;

//--------------------------------------------------------------------

void filter_monox(
TString inputFile    = "/data/blue/ceballos/monox/ntuples/ttbar.root"
) {   
#ifdef __CINT__
  gROOT->ProcessLine( ".O0" ); // turn off optimization in CINT
#endif

  vector<TString> samples;
  samples.push_back(inputFile.Data());

  // --------------------------------------------------------------------------------------------------
  const unsigned int nsamples = samples.size();
  
  TString *allEvts = new TString("hDAllEvents");
  TString *dirFwk  = new TString("AnaFwkMod");

  for( unsigned int i = 0 ; i < nsamples ; ++i ){

    TFile fif(samples.at(i).Data());
    if(fif.GetDirectory(dirFwk->Data())) fif.cd(dirFwk->Data());
    TH1D *hAllEvts = (TH1D*) gROOT->FindObject(allEvts->Data());

    TChain *ch = new TChain("tree");
    ch -> Add( Form("%s" , samples.at(i).Data()) );

    MitDMSTree signal;
    signal.LoadTree(samples.at(i).Data(),"DMSTree");
    signal.InitTree(0);

    TString ofn(samples.at(i).Data());
    ofn.ReplaceAll(".root","_filter.root");
    TFile *out = TFile::Open( Form("%s" , ofn.Data() ) ,"RECREATE" );
    out->cd();
    TTree *clone = signal.tree_->CloneTree(0);
  
    TStopwatch sw;
    sw.Start();

    int npass   = 0;

    for (Long64_t ievt=0; ievt<signal.tree_->GetEntries();ievt++) {
 
      if (ievt%1000000 == 0) std::cout << "--- ... Processing event: " << ievt << " of " << signal.tree_->GetEntries() << std::endl;

      signal.tree_->GetEntry(ievt);
      
      LorentzVector leptonSystem = signal.lep1_;
      if(signal.nlep_ >= 2) leptonSystem = signal.lep1_+signal.lep2_;

      bool passFjet[2] = {signal.fjet1_.Pt() > 250 && abs(signal.fjet1_.Eta()) < 2.5, 
                          signal.fjet2_.Pt() > 250 && abs(signal.fjet2_.Eta()) < 2.5};
      bool filter = passFjet[0] || passFjet[1];

      if(!filter) continue;

      bool leptonCuts = true;
      if     (signal.nlep_ == 0) {}
      else if(signal.nlep_ >= 1 && signal.nlep_ <= 2) {
        leptonCuts = leptonCuts && ((passFjet[0] && deltaR(signal.fjet1_.phi(),signal.fjet1_.eta(),leptonSystem.phi(),leptonSystem.eta()) > 0.3) ||
	                            (passFjet[1] && deltaR(signal.fjet2_.phi(),signal.fjet2_.eta(),leptonSystem.phi(),leptonSystem.eta()) > 0.3));
      } else {
        leptonCuts = true;
      }

      if(!leptonCuts) continue;

      npass++;

      clone->Fill();

    } // End main loop

    std::cout << npass << " events passing selection" << std::endl;
 
    // Get elapsed time
    sw.Stop();
    std::cout << "--- End of event loop: "; sw.Print();

    // --- write output baby

    clone->Write();
    hAllEvts->Write();
    out->Close();
    
    std::cout << "==> FilterApplication is done to sample " << ofn.Data() << endl << std::endl;
  } 
}
