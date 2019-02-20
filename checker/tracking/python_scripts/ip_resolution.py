import os, sys
import ROOT
import numpy as np
from array import array

sys.path.append('../../')
from plotting.LHCbStyle import *
setLHCbStyle()

edges = np.array([0.0, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
bins = zip(edges[:-1], edges[1:])
centers = array('d', 0.5 * (edges[:-1] + edges[1:]))
widths = array('d', 0.5 * (edges[1:] - edges[:-1]))


def ipRes3D(tree, var, ptBin):
    th1 = ROOT.TH1F('h', 'h', 100, 0, 0.1)
    tree.Draw(
        var + '>>h',
        'ghost==0 && 1000./best_pt>={} && 1000./best_pt<{}'.format(
            ptBin[0], ptBin[1]), 'goff')
    return th1.GetMean(), th1.GetStdDev() / np.sqrt(th1.Integral())


def ipResFit(tree, var, ptBin):
    th1 = ROOT.TH1F('h', 'h', 100, -0.1, 0.1)
    tree.Draw(
        var + '>>h',
        'ghost==0 && 1000./best_pt>={} && 1000./best_pt<{}'.format(
            ptBin[0], ptBin[1]), 'goff')
    f = ROOT.TF1('f', 'gaus', -0.1, 0.1)
    th1.Fit(f, 'R')
    return th1.GetFunction('f').GetParameter(2), th1.GetFunction(
        'f').GetParError(2)


def makeGraph(tree, var, resFunc):
    res = np.array([resFunc(tree, var, ptBin) for ptBin in bins])
    g = ROOT.TGraphErrors(
        len(centers), centers, array('d', 1000. * res[:, 0]), widths,
        array('d', 1000. * res[:, 1]))
    g.GetHistogram().GetXaxis().SetTitle('1/#it{p}_{T} [#it{c}/GeV]')
    g.GetHistogram().GetYaxis().SetTitle('IP resolution [#mum]')
    g.GetHistogram().GetXaxis().SetRangeUser(edges[0], edges[-1])
    g.SetName(var + '_res')
    return g


if __name__ == '__main__':
    ipVars = ['kalman_ipx', 'kalman_ipy', 'velo_ipx', 'velo_ipy']
    ipVars3D = ['kalman_ip', 'velo_ip']
    inFile = ROOT.TFile('../../../output/KalmanIPCheckerOutput.root')
    inTree = inFile.Get('kalman_ip_tree')
    outFile = ROOT.TFile('ip_resolution.root', 'recreate')
    graphs = [makeGraph(inTree, var, ipResFit) for var in ipVars]
    graphs += [makeGraph(inTree, var, ipRes3D) for var in ipVars3D]
    for graph in graphs:
        graph.Write()
    outFile.Write()
    outFile.Close()
