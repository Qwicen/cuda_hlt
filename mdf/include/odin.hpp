
#ifndef ODIN_H
#define ODIN_H 1

namespace LHCb {
struct ODIN final {
   enum Data{ RunNumber = 0,
              EventType,
              OrbitNumber,
              L0EventIDHi,
              L0EventIDLo,
              GPSTimeHi,
              GPSTimeLo,
              Word7,
              Word8,
              TriggerConfigurationKey
   };

   enum EventTypeBitsEnum{ EventTypeBits = 0,
                           CalibrationStepBits = 16
   };

   enum EventTypeMasks{ EventTypeMask       = 0x0000FFFF,
                        CalibrationStepMask = 0xFFFF0000,
                        FlaggingModeMask    = 0x00008000
   };

   unsigned int run_number;
   unsigned int event_type;
   unsigned int orbit_number;
   unsigned long long event_number;
   unsigned int version;
   unsigned int calibration_step;
   unsigned int tck;
};
}

#endif
