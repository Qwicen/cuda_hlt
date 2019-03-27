import os, sys
import ROOT
import numpy as np
from array import array

sys.path.append('../../')
from plotting.LHCbStyle import *
setLHCbStyle()

if __name__ == '__main__':
    inFile = ROOT.TFile('../../../output/KalmanIPCheckerOutput.root')
    inTree = inFile.Get('kalman_ip_tree')
    c1 = ROOT.TCanvas('c1', 'c1')
    hKalman = ROOT.TH1F('hKalman', 'hKalman', 20, 0, 2)
    hVelo = ROOT.TH1F('hVelo', 'hVelo', 20, 0, 2)
    inTree.Draw('kalman_docaz>>hKalman', 'ghost==0', 'goff')
    inTree.Draw('velo_docaz>>hVelo', 'ghost==0', 'goff')
    hKalman.SetLineColor(ROOT.kCyan + 1)
    hKalman.SetMarkerColor(ROOT.kCyan + 1)
    hVelo.SetLineColor(ROOT.kBlack)
    hVelo.SetMarkerColor(ROOT.kBlack)
    hKalman.Draw('E')
    hVelo.Draw('E same')
    hKalman.GetXaxis().SetTitle('DOCA#it{z} [mm]')
    legend = ROOT.TLegend(0.7, 0.92, 0.9, 0.77)
    legend.AddEntry(hVelo, 'VELO', 'lp')
    legend.AddEntry(hKalman, 'Kalman', 'lp')
    legend.SetFillStyle(0)
    legend.Draw('same')
    c1.SetLogy(True)
    c1.SaveAs('kalman_docaz.pdf')
