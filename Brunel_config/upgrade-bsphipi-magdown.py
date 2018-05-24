#-- GAUDI jobOptions generated on Mon Oct 30 12:21:03 2017
#-- Contains event types : 
#--   13104012 - 40 files - 45062 events - 162.12 GBytes


#--  Extra information about the data processing phases:


#--  Processing Pass Step-132412 

#--  StepId : 132412 
#--  StepName : Digi14b-Upgrade for Upgrade studies with spillover - 2017 Baseline NoRichSpillover - xdigi 
#--  ApplicationName : Boole 
#--  ApplicationVersion : v31r3 
#--  OptionFiles : $APPCONFIGOPTS/Boole/Default.py;$APPCONFIGOPTS/Boole/Boole-Upgrade-Baseline-20150522.py;$APPCONFIGOPTS/Boole/EnableSpillover.py;$APPCONFIGOPTS/Boole/Upgrade-RichMaPMT-NoSpilloverDigi.py;$APPCONFIGOPTS/Boole/xdigi.py 
#--  DDDB : dddb-20171010 
#--  CONDDB : sim-20170301-vc-md100 
#--  ExtraPackages : AppConfig.v3r338 
#--  Visible : N 

from Gaudi.Configuration import * 
from GaudiConf import IOHelper
IOHelper('ROOT').inputFiles(['LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000003_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000004_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000005_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000006_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000007_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000008_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000010_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000014_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000015_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000016_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000017_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000018_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000019_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000020_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000021_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000023_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000025_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000027_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000030_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000032_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000034_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000037_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000038_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000039_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000041_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000042_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000043_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000046_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000047_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000048_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000049_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000050_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000052_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000053_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000054_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000060_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000064_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000065_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000066_1.xdigi',
'LFN:/lhcb/MC/Upgrade/XDIGI/00067195/0000/00067195_00000068_1.xdigi'
], clear=True)
FileCatalog().Catalogs += [ 'xmlcatalog_file:upgrade-bsphipi-magdown.xml' ]
