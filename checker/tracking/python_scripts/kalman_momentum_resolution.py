import os, sys
import argparse
import ROOT
import numpy as np
from array import array

sys.path.append('../../')
from plotting.LHCbStyle import *
from plotting.Legend import *
setLHCbStyle()

edges = np.array([2000.,5000.,10000.,20000.,30000.,40000.,60000.,100000.])
bins = zip(edges[:-1],edges[1:])
centers = 0.5*(edges[:-1]+edges[1:])
widths = 0.5*(edges[1:]-edges[:-1])

def pResFit(tree, pvar, bin):
    th1 = ROOT.TH1F('h','h',100,-0.05,0.05)
    tree.Draw('('+pvar+'-mcp_p)/mcp_p>>h',
              'ghost==0 && mcp_p>={} && mcp_p<{}'.format(bin[0],bin[1]),
              'goff')
    f = ROOT.TF1('f','gaus',-0.05,0.05)
    th1.Fit(f,'R')
    return th1.GetFunction('f').GetParameter(2), th1.GetFunction('f').GetParError(2)

if __name__ == '__main__':
    inFile = ROOT.TFile('../../../output/KalmanIPCheckerOutput.root')
    inTree = inFile.Get('kalman_ip_tree')
    firstRes = np.array([pResFit(inTree,'1./first_qop',bin) for bin in bins])
    bestRes = np.array([pResFit(inTree,'1./best_qop',bin) for bin in bins])

    c1 = ROOT.TCanvas('c1','c1')
    centers = centers/1000.
    widths = widths/1000.
    firstG = ROOT.TGraphErrors(len(centers),
                               array('d',centers),
                               array('d',firstRes[:,0]),
                               array('d',widths),
                               array('d',firstRes[:,1]))
    bestG = ROOT.TGraphErrors(len(centers),
                              array('d',centers),
                              array('d',bestRes[:,0]),
                              array('d',widths),
                              array('d',bestRes[:,1]))
    firstG.SetMarkerColor(ROOT.kBlack)
    firstG.SetLineColor(ROOT.kBlack)
    bestG.SetMarkerColor(ROOT.kCyan+1)
    bestG.SetLineColor(ROOT.kCyan+1)
    mg = ROOT.TMultiGraph()
    mg.Add(firstG)
    mg.Add(bestG)
    mg.Draw('ap')
    mg.GetHistogram().GetXaxis().SetTitle('#it{p} [GeV/#it{c}^{2}]')
    mg.GetHistogram().GetYaxis().SetTitle('#sigma_{#it{p}}/#it{p}')
    mg.GetHistogram().GetYaxis().SetRangeUser(0,0.015);
    mg.GetYaxis().SetTitleOffset(1.1)
    mg.GetHistogram().GetXaxis().SetRangeUser(edges[0]/1000.,edges[-1]/1000.)
    legend = ROOT.TLegend(0.67,0.92,0.95,0.72)
    legend.AddEntry(firstG,'Initial')
    legend.AddEntry(bestG,'Kalman')
    legend.SetFillStyle(0)
    legend.Draw('same')
    c1.SaveAs('kalman_momentum_resolution.pdf')
