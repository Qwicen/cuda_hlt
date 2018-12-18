#!/usr/bin/python

# Script for obtaining the momentum resolution versus momentum
# from a 2D histogram of dp versus p
#
# author: Dorothea vom Bruch (dorothea.vom.bruch@cern.ch)
# date:   12/2018
# 

import os,sys 
import argparse 
import ROOT
from ROOT import *
from ROOT import gStyle
from ROOT import gROOT
from ROOT import TStyle
from ROOT import gPad

from LHCbStyle import *

def getTrackers() :
    return ["Velo", "Upstream", "Forward"]
    

f = ROOT.TFile.Open("../../../output/PrCheckerPlots.root", "read")
outputfile = ROOT.TFile( "momentum_resolution.root", "recreate" )

for tracker in trackers :
    outputfile.cd()
    trackerDir = outputfile.mkdir(tracker)
    trackerDir.cd()

    # get histogram
    name = tracker + "/dp_vs_p"
    histo = f.Get(name)
