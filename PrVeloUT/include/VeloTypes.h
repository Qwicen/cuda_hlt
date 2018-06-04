struct HitBase { // 4 * 4 = 16 B
    float x;
    float y;
    float z;
      
    HitBase(){}
    HitBase(
      const float _x,
      const float _y,
      const float _z
    ) : x(_x), y(_y), z(_z) {}
};

template <>
struct Hit : public HitBase {
    uint32_t LHCbID;
    Hit(){}
    Hit(
      const float _x,
      const float _y,
      const float _z,
      const uint32_t _LHCbID
    ) : HitBase( _x, _y, _z ), LHCbID( _LHCbID ) {}
};

template <>
struct Hit <false> : public HitBase {
     Hit(){}
     Hit(
       const float _x,
       const float _y,
       const float _z
    ) : HitBase( _x, _y, _z) {}
};

/* Structure containing indices to hits within hit array */
struct TrackHits { // 4 + 26 * 4 = 116 B
  unsigned short hitsNum;
  unsigned short hits[VeloTracking::max_track_size];

  TrackHits(){}
  TrackHits(
    const unsigned short _hitsNum,
    const unsigned short _h0,
    const unsigned short _h1,
    const unsigned short _h2
  ) : hitsNum(_hitsNum) {
    hits[0] = _h0;
    hits[1] = _h1;
    hits[2] = _h2;
  }
};

/* Structure to save final track
   Contains information needed later on in the HLT chain
   and / or for truth matching */
template <bool MCCheck>   
struct Track { // 4 + 26 * 16 = 420 B
  unsigned short hitsNum;
  Hit <MCCheck> hits[VeloTracking::max_track_size];
  
  Track(){
    hitsNum = 0;
  }
 
  void addHit( Hit <MCCheck> _h ){
    hits[ hitsNum ] = _h;
    hitsNum++;
  }
}; 

/**
 * @brief A simplified state for the Velo
 *        
 *        {x, y, tx, ty, 0}
 *        
 *        associated with a simplified covariance
 *        since we do two fits (one in X, one in Y)
 *
 *        c00 0.f c20 0.f 0.f
 *            c11 0.f c31 0.f
 *                c22 0.f 0.f
 *                    c33 0.f
 *                        0.f
 */
struct VeloState { // 48 B
  float x, y, tx, ty;
  float c00, c20, c22, c11, c31, c33;
  float chi2;
  float z;
};

