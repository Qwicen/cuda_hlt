from Gaudi.Configuration import *
from Brunel.Configuration import Brunel
from Configurables import ( LHCbConfigurableUser, LHCbApp,
                             L0Conf,
                            TrackSys)
from GaudiConf import IOHelper

#new MC in tmp/
DDDBtag    = "dddb-20171010"
CondDBtag  = "sim-20170301-vc-md100"


from Configurables import Brunel
from Gaudi.Configuration import *
from GaudiKernel.SystemOfUnits import mm
from GaudiKernel.SystemOfUnits import GeV
from GaudiConf import IOHelper
from GaudiKernel.SystemOfUnits import mm
from GaudiKernel.SystemOfUnits import GeV

Evts_to_Run = 10 # set to -1 to process all

mbrunel = Brunel( DataType = "Upgrade",
                  EvtMax = Evts_to_Run,
                  #SkipEvents = 1,
                  PrintFreq = 1,
                  WithMC = True,
                  Simulation = True,
                  OutputType = "None",
                  DDDBtag    = "dddb-20170301",
                  CondDBtag  = "sim-20170301-vc-md100",
                  MainSequence = ['ProcessPhase/Reco'],
                  RecoSequence = ["Decoding", "TrFast"],
                  Detectors = ["VP","UT","FT"],
                  InputType = "DIGI"
                  )
 
TrackSys().TrackingSequence = ["TrFast"]
TrackSys().TrackTypes       = ["Velo","Upstream","Forward"]
mbrunel.MainSequence += ['ProcessPhase/MCLinks',  'ProcessPhase/Check']
#L0Conf.EnsureKnownTKC = False

from Configurables import PrEventDumper  
EventDumper = PrEventDumper("PrEventDumper")
EventDumper.containingFolder = "velopix_MC"
 
def ConfGaudiSeq():
    from Configurables import PrEventDumper
    GaudiSequencer("BrunelSequencer").Members.append(PrEventDumper())

from GaudiKernel.Configurable import appendPostConfigAction
appendPostConfigAction(ConfGaudiSeq)


