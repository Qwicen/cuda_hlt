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

    float m_cos;
    float m_dxDy;    ///< The dx/dy value
    float m_weight;  ///< The hit weight (1/error)
    float m_xAtYEq0; ///< The value of x at the point y=0
    float m_yBegin;  ///< The y value at the start point of the line
    float m_yEnd;    ///< The y value at the end point of the line
    float m_zAtYEq0; ///< The value of z at the point y=0
    float m_x;

    float m_second_x;
    float m_second_z;

    Hit(){}
    Hit(
      const float _x,
      const float _y,
      const float _z,
      const uint32_t _LHCbID
    ) : HitBase( _x, _y, _z ), LHCbID( _LHCbID ) {}

    inline float cos() const { return m_cos; }
    inline float cosT() const { return ( fabs( m_xAtYEq0 ) < 1.0E-9 ) ? 1. / std::sqrt( 1 + m_dxDy * m_dxDy ) : cos(); }
    inline float dxDy() const { return m_dxDy; }
    // inline bool highThreshold() const { return m_cluster.highThreshold(); }
    inline bool isYCompatible( const float y, const float tol ) const { return yMin() - tol <= y && y <= yMax() + tol; }
    inline bool isNotYCompatible( const float y, const float tol ) const { return yMin() - tol > y || y > yMax() + tol; }
    // inline LHCb::LHCbID lhcbID() const { return LHCb::LHCbID( m_cluster.channelID() ); }
    // inline int planeCode() const { return 2 * ( m_cluster.station() - 1 ) + ( m_cluster.layer() - 1 ) % 2; }
    inline float sinT() const { return tanT() * cosT(); }
    // inline int size() const { return m_cluster.pseudoSize(); }
    inline float tanT() const { return -m_dxDy; }
    inline float weight() const { return m_weight * m_weight; }
    inline float xAt( const float globalY ) const { return m_xAtYEq0 + globalY * m_dxDy; }
    inline float xAtYEq0() const { return m_xAtYEq0; }
    inline float xAtYMid() const { return m_x; }
    inline float xMax() const { return std::max( xAt( yBegin() ), xAt( yEnd() ) ); }
    inline float xMin() const { return std::min( xAt( yBegin() ), xAt( yEnd() ) ); }
    inline float xT() const { return cos(); }
    inline float yBegin() const { return m_yBegin; }
    inline float yEnd() const { return m_yEnd; }
    inline float yMax() const { return std::max( yBegin(), yEnd() ); }
    inline float yMid() const { return 0.5 * ( yBegin() + yEnd() ); }
    inline float yMin() const { return std::min( yBegin(), yEnd() ); }
    inline float zAtYEq0() const { return m_zAtYEq0; }
};

typedef std::vector<Hit> Hits;

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

