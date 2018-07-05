typedef float data_t;
const data_t WEIGHT = 3966.94;
const float FACTOR = 1.;

struct hit_t {
  float x;
  float y;
  float z;
  int   tid;
  int module;
  bool used = false;
  
   //compare two hits by track ID
  
  bool operator==(int tid_) {
    return tid == tid_;
  } 
};



struct track_t {
  std::vector< const hit_t * > hits;
  std::vector< int > hit_ids;
  //track od
  int key;

};

struct state_t {
  data_t x = 0;
  data_t y = 0;
  data_t z = 0;
  data_t tx = 0;
  data_t ty = 0;
  
  data_t covXX;
  data_t covYY;
  data_t covXTx;
  data_t covYTy;
  data_t covTxTx;
  data_t covTyTy;
  
  data_t chi2 = 0.;
  data_t chi2_x = 0.;
  data_t chi2_y = 0.;
  int key;
};