#ifndef MDFHEADER
#define MDFHEADER

#include <stdexcept>
#define DAQ_ERR_BANK_VERSION 0
#define DAQ_STATUS_BANK      16
#define DAQ_PROCERR_HEADER   32
#define DAQ_PROCERR_BANK     33
#define DAQ_FILEID_BANK      255

#define MDFHEADER_ALIGNED(x) x __attribute__((__packed__))

/*
 *   LHCb namespace
 */
namespace LHCb    {

  /** @struct MDFHeader  MDFHeader.h  MDF/MDFHeader.h
    *
    * Structure describing the header structure preceding each
    * event buffer in MDF files.
    *
    * Known versions:
    * 0   : VELO testbeam  [early version]
    * 1   : RICH/MUON/OTR test beam
    * 2   : Empty specific header
    * 3   : New version (like 1, but with data type)
    *
    * Known data types:
    * 1 BODY_TYPE_BANKS
    * 2 BODY_TYPE_MEP
    *
    * Caution:
    * The data member need to be aligned in a way that the compiler
    * does not inject additional padding !
    *
    * @author  M.Frank
    * @version 1.0
    *
    */
  MDFHEADER_ALIGNED(class) MDFHeader  {
  public:
    enum { BODY_TYPE_BANKS=1, BODY_TYPE_MEP=2 };

    MDFHEADER_ALIGNED(struct) HeaderTriggerMask  {
      /// Trigger mask used for event selection
      unsigned int   m_trMask[4];
      HeaderTriggerMask() {
        m_trMask[0] = m_trMask[1] = m_trMask[2] = m_trMask[3] = 0;
      }
      /// Accessor: Number of bits in the trigger mask
      unsigned int  maskBits() const         { return sizeof(m_trMask)*8;        }
      /// Accessor: trigger mask
      const unsigned int* triggerMask() const{ return m_trMask;                  }
      /// Update the trigger mask of the event
      void setTriggerMask(const unsigned int* mask){
        m_trMask[0] = mask[0];
        m_trMask[1] = mask[1];
        m_trMask[2] = mask[2];
        m_trMask[3] = mask[3];
      }
    };

    MDFHEADER_ALIGNED(struct) Header1  : public HeaderTriggerMask  {
      /// Run number
      unsigned int   m_runNumber = 0;
      /// Orbit counter
      unsigned int   m_orbitCount = 0;
      /// Bunch identifier
      unsigned int   m_bunchID = 0;
      /// Set run number
      void setRunNumber(unsigned int runno)   { m_runNumber  = runno;   }
      /// Set orbit counter
      void setOrbitNumber(unsigned int orbno) { m_orbitCount = orbno;   }
      /// Set bunch identifier
      void setBunchID(unsigned int bid)       { m_bunchID    = bid;     }
      /// Access run number
      unsigned int runNumber() const          { return m_runNumber;     }
      /// Access run number
      unsigned int orbitNumber() const        { return m_orbitCount;    }
      /// Access run number
      unsigned int bunchID()   const          { return m_bunchID;       }
    };


    /// Data member indicating the size of the event
    unsigned int   m_size[3];
    /// Optional checksum over the event data (if 0, no checksum was calculated)
    unsigned int   m_checkSum;
    /// Identifier of the compression algorithm used to compress the data buffer
    unsigned char  m_compression;
    /// Header type: split into { version:4, length:4 } for possible future upgrade
    unsigned char  m_hdr;
    /// Data type
    unsigned char  m_dataType;
    /// Spare
    Header1  m_subHeader;


  public:
    static unsigned int sizeOf(int hdr_type)  {
      switch(hdr_type)  {
        case 3:
           return sizeof(MDFHeader)+sizeof(Header1);
        default:
          throw std::runtime_error("Unknown MDF header type!");
      }
    }
    /// Default constructor
    MDFHeader() : m_checkSum(0), m_compression(0), m_hdr(0), m_dataType(0)
    {
      m_size[0] = m_size[1] = m_size[2] = 0;
      setSubheaderLength(0);
    }
    /// Default destructor
    ~MDFHeader()  {}
    /// Access record size
    unsigned int  recordSize()  const      { return m_size[0];                 }
    /// Accessor: event size
    unsigned int  size() const
    { return m_size[0]-sizeOf(headerVersion());                                }
    /// Update event size
    void setSize(unsigned int val)
    { m_size[0]=m_size[1]=m_size[2]=val+sizeOf(headerVersion());               }
    /// For checks: return 0th. size word
    unsigned int size0() const             { return m_size[0];                 }
    /// For checks: return 1rst. size word
    unsigned int size1() const             { return m_size[1];                 }
    /// For checks: return 2nd. size word
    unsigned int size2() const             { return m_size[2];                 }
    /// For special stuff: modify 3rd. size word by hand
    void setSize2(unsigned int val)        { m_size[2] = val;                  }
    /// Accessor: checksum of the event data
    unsigned int  checkSum() const         { return m_checkSum;                }
    /// Update checksum of the event data
    void setChecksum(unsigned int val)     { m_checkSum = val;                 }
    /// Accessor: Identifier of the compression method
    unsigned char compression() const      { return m_compression;             }
    /// Update the identifier of the compression method
    void setCompression(unsigned int val)  { m_compression=(unsigned char)val; }
    /// Accessor: length of the event header
    unsigned int subheaderLength() const   { return (m_hdr&0x0F)*sizeof(int);  }
    /// Update the length of the event header
    void setSubheaderLength(unsigned int l)  {
       l = (l%sizeof(int)) ? (l/sizeof(int)) + 1 : l/sizeof(int);
       m_hdr = (unsigned char)((0xF0&m_hdr) + (0x0F&l));
    }
    /// Accessor: version of the event header
    unsigned int  headerVersion() const    { return m_hdr>>4;                  }
    /// Update the version of the event header
    void setHeaderVersion(unsigned int vsn)
    {  m_hdr = (unsigned char)(((vsn<<4)+(m_hdr&0xF))&0xFF);                   }
    /// Accessor: hdr field
    unsigned char hdr() const              { return m_hdr;                     }
    /// Update hdr field
    void setHdr(unsigned char val)         { m_hdr = val;                      }
    /// Accessor: event type identifier
    unsigned char dataType() const         { return m_dataType;                }
    /// Update the event type
    void setDataType(unsigned char val)    { m_dataType = val;                 }
    /// Access to data payload (Header MUST be initialized)
    char* data() {  return ((char*)this)+sizeOf(headerVersion());              }
    /// Access to data payload (Header MUST be initialized)
    const char* data() const {  return ((char*)this)+sizeOf(headerVersion());  }

    /// Access to sub-headers
    Header1 subHeader()                  { return m_subHeader;      }
  };
}    // End namespace LHCb

#undef MDFHEADER_ALIGNED
#endif // EVENT_MDFHEADER
