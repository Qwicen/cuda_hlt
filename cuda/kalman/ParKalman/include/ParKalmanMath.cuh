#pragma once

namespace ParKalmanFilter {

  //----------------------------------------------------------------------
  // Template declaration
  template<bool _sym, int _size>
  struct SquareMatrix;

  //----------------------------------------------------------------------
  // Non-symmetric, square matrix.
  template<int _size>
  struct SquareMatrix<false, _size> {
    int size;
    double vals[_size*_size];
    __host__ __device__ SquareMatrix(){
      size = _size;
      for(int i=0; i<_size*_size; i++)
        vals[i] = 0.;
    }
    __host__ __device__ SquareMatrix(const double init_vals[_size*_size]){
      size = _size;
      for(int i=0; i<_size*_size; i++)
        vals[i] = init_vals[i];
    }
    __host__ __device__ void SetElements(const double init_vals[_size*_size]){
      for(int i=0; i<size*size; i++)
        vals[i] = init_vals[i];
    }
    __host__ __device__ double& operator()(int i, int j){
      return vals[i*_size+j];
    }
    __host__ __device__ const double& operator()(int i, int j) const {
      return vals[i*_size+j];
    }
    __host__ __device__ SquareMatrix<false,_size> T(){
      SquareMatrix<false,_size> ret;
      for(int i=0; i<_size; i++)
        for(int j=0; j<_size; j++)
          ret(i,j) = vals[j*_size+i];
      return ret;
    }
    __host__ __device__ SquareMatrix<false,_size> T() const {
      SquareMatrix<false,_size> ret;
      for(int i=0; i<_size; i++)
        for(int j=0; j<_size; j++)
          ret(i,j) = vals[j*_size+i];
      return ret;
    }
      
  };

  //----------------------------------------------------------------------
  // Symmetric matrix.
  template<int _size>
  struct SquareMatrix<true, _size> {
    int size;
    double vals[_size*(_size+1)/2];
    __host__ __device__ SquareMatrix(){
      size = _size;
      for(int i=0; i<_size*(_size+1)/2; i++)
        vals[i] = 0;
    }
    __host__ __device__ SquareMatrix(double init_vals[_size*(_size+1)/2]){
      size = _size;
      for(int i=0; i<_size*(_size+1)/2; i++)
        vals[i] = init_vals[i];
    }
    __host__ __device__ void SetElements(double init_vals[_size*(_size+1)/2]){
      for(int i=0; i<size*(size+1)/2; i++)
        vals[i] = init_vals[i];
    }
    __host__ __device__ double& operator()(int i, int j){
      if(i>j){ int tmp=i; i=j; j=tmp; }
      int idx=i;
      while(j>0){ idx+=j; j-=1; }
      return vals[idx];
    }
    __host__ __device__ const double& operator()(int i, int j) const {
      if(i>j){ int tmp=i; i=j; j=tmp; }
      int idx=i;
      while(j>0){ idx+=j; j-=1; }
      return vals[idx];      
    }
  };

  //----------------------------------------------------------------------
  // Vector.
  template<int _size>
  struct Vector {
    int size;
    double vals[_size];
        
    __host__ __device__ Vector();
    __host__ __device__ Vector(double init_vals[_size]);
    __host__ __device__ double& operator()(int i){
      return vals[i];
    }
    __host__ __device__ const double& operator()(int i) const {
      return vals[i];
    }
    __host__ __device__ double& operator[](int i){
      return vals[i];
    }
    __host__ __device__ const double& operator[](int i) const {
      return vals[i];
    }
    
  };

  template<int _size>
  __host__ __device__ Vector<_size>::Vector(){
    size = _size;
    for(int i=0; i<_size; i++)
      vals[i]=0;    
  }

  template<int _size>
  __host__ __device__ Vector<_size>::Vector(double init_vals[_size]){
    size = _size;
    for(int i=0; i<_size; i++)
      vals[i] = init_vals[i];
  }


  //----------------------------------------------------------------------
  // Invert a square matrix
  template<bool _sym, int _size>
  __host__ __device__ SquareMatrix<_sym,_size> inverse(
    const SquareMatrix<_sym,_size> &A
  ){
    SquareMatrix<_sym,_size> Ainv;
    SquareMatrix<false,_size> ut;
    SquareMatrix<false,_size> lt;

    // Decompose
    for(int i=0; i<_size; i++){
      // Upper triangular
      for(int k=i; k<_size; k++){
        double sum=0;
        for(int j=0; j<i; j++)
          sum += lt(i,j)*ut(j,k);
        ut(i,k) = A(i,k) - sum;
      }
      // Lower triangular
      for(int k=i; k<_size; k++){
        if(i==k) lt(i,i) = 1;
        else{
          double sum=0;
          for(int j=0; j<i; j++)
            sum += lt(k,j)*ut(j,i);
          lt(k,i) = (A(k,i)-sum)/ut(i,i);
        }
      }    
    }

    // Invert triangular matrices.
    SquareMatrix<false,_size> utinv;
    SquareMatrix<false,_size> ltinv;
    // Set diagonals.
    for(int i=0; i<_size; i++){
      utinv(i,i) = 1./ut(i,i);
      ltinv(i,i) = 1./lt(i,i);
    }
    // Off-diagonal.
    for(int i=0; i<_size; i++){    
      // Upper.
      for(int off=1; off<_size-i; off++){
        int j=i+off;
        double utval=0;
        double ltval=0;
        for(int k=i; k<=j; k++){
          utval -= utinv(i,k)*ut(k,j);
          ltval -= ltinv(k,i)*lt(j,k);
        }
        utval /= ut(j,j);
        ltval /= lt(j,j);
        utinv(i,j) = utval;
        ltinv(j,i) = ltval;
      }    
    }

    // Make the inverse.
    multiplySquareBySquare(utinv,ltinv,Ainv);
    return Ainv;
  }

  //----------------------------------------------------------------------
  // Multiply two square matrices. C=AxB.
  template<bool _symA, bool _symB, bool _symC, int _size>
  __device__ __host__ void multiplySquareBySquare(const SquareMatrix<_symA,_size> &A,
                                         const SquareMatrix<_symB,_size> &B,
                                         SquareMatrix<_symC,_size> &C){
    for(int i=0; i<_size; i++){
      for(int j=0; j<_size; j++){
        double sum=0;
        for(int k=0; k<_size; k++) sum += A(i,k)*B(k,j);
        C(i,j) = sum;
      }
    }
  }

  //----------------------------------------------------------------------
  // Operator * Multiply two square matrices. C=A*B.
  template<bool _symA, bool _symB, int _size>
  __host__ __device__ SquareMatrix<false,_size> operator*(const SquareMatrix<_symA, _size> &A, const SquareMatrix<_symB,_size> &B) {
    SquareMatrix<false,_size> C;
    for(int i=0; i<_size; i++){
      for(int j=0; j<_size; j++){
        double sum=0;
        for(int k=0; k<_size; k++) sum += A(i,k)*B(k,j);
        C(i,j) = sum;
      }
    }
    return C;
  }

  //----------------------------------------------------------------------
  // Operator + Add two square matrices. C=A+B.
  template<bool _symA, bool _symB, int _size>
  __host__ __device__ SquareMatrix<false,_size> operator+(const SquareMatrix<_symA, _size> &A, const SquareMatrix<_symB,_size> &B) {
    SquareMatrix<false,_size> C;
    for(int i=0; i<_size; i++){
      for(int j=0; j<_size; j++){
        C(i,j) = A(i,j) + B(i,j);
      }  
    }
    return C;
  }

  template<int _size>
  __host__ __device__ SquareMatrix<true,_size> operator+(const SquareMatrix<true, _size> &A, const SquareMatrix<true,_size> &B) {
    double res_vals[_size*(_size+1)/2];
    for(int i=0; i<(_size*(_size+1)/2); i++){
      res_vals[i] = A.vals[i] + B.vals[i];
    }
    return SquareMatrix<true, _size>(res_vals);
  }

  //----------------------------------------------------------------------
  // Operator - Subtract a square matrix from another. C=A-B.
  template<bool _symA, bool _symB, int _size>
  __host__ __device__ SquareMatrix<false,_size> operator-(const SquareMatrix<_symA,_size> &A,
                                                          const SquareMatrix<_symB,_size> &B) {
    SquareMatrix<false,_size> C;
    for(int i=0; i<_size; i++){
      for(int j=0; j<_size; j++){
        C(i,j) = A(i,j) - B(i,j);
      }
    }
    return C;
  }

  template<int _size>
  __host__ __device__ SquareMatrix<true,_size> operator-(const SquareMatrix<true, _size> &A, const SquareMatrix<true,_size> &B) {
    double res_vals[_size*(_size+1)/2];
    for(int i=0; i<(_size*(_size+1)/2); i++){
      res_vals[i] = A.vals[i] - B.vals[i];
    }
    return SquareMatrix<true, _size>(res_vals);
  }
  
  //----------------------------------------------------------------------
  // Operator + Add two vectors. C=A+B.
  template<int _size>
  __host__ __device__ Vector<_size> operator+(const Vector<_size> &A, const Vector<_size> &B) {
    Vector<_size> C;
    for(int i=0; i<_size; i++){
      C(i) = A(i) + B(i);
    }
    return C;
  }

  //----------------------------------------------------------------------
  // Operator - Subtract a vector from another. C=A-B.
  template<int _size>
  __host__ __device__ Vector<_size> operator-(const Vector<_size> &A, const Vector<_size> &B) {
    Vector<_size> C;
    for(int i=0; i<_size; i++){
      C(i) = A(i) - B(i);
    }
    return C;
  }

  //----------------------------------------------------------------------
  // Operator * Multiply square matrix by vector. C=A*B.
  template<bool _symA, int _size>
  __host__ __device__ Vector<_size> operator*(const SquareMatrix<_symA, _size> &A, const Vector<_size> &B) {
    Vector<_size> C;
    for(int i=0; i<_size; i++){
      double sum=0;
      for(int j=0; j<_size; j++) sum += A(i,j)*B(j);
      C(i) = sum;
    }
    return C;
  }


  //----------------------------------------------------------------------
  // Operator for scalar multiplication of vectors.
  template<int _size>
  __host__ __device__ Vector<_size> operator*(const double &a, const Vector<_size> v){
    Vector<_size> u;
    for(int i=0; i<u.size; i++)
      u(i) = v(i)*a;
    return u;
  }

  //----------------------------------------------------------------------
  // Operator for scalar multiplication of square matrices.
  template<bool _sym, int _size>
  __host__ __device__ SquareMatrix<_sym,_size> operator*(const double &a, const SquareMatrix<_sym,_size> M){
    SquareMatrix<_sym,_size> aM;
    for(int i=0; i<_size; i++)
      for(int j=0; j<_size; j++)
        aM(i,j) = M(i,j)*a;
    return aM;
  }

  //----------------------------------------------------------------------
  // Operator for scalar division of vectors.
  template<int _size>
  __host__ __device__ Vector<_size> operator/(const Vector<_size> v, const double &a){
    Vector<_size> u;
    for(int i=0; i<u.size; i++)
      u(i) = v(i)/a;
    return u;
  }

  //----------------------------------------------------------------------
  // Operator for scalar multiplication of square matrices.
  template<bool _sym, int _size>
  __host__ __device__ SquareMatrix<_sym,_size> operator/(const SquareMatrix<_sym,_size> M, const double &a){
    SquareMatrix<_sym,_size> Moa;
    for(int i=0; i<_size; i++)
      for(int j=0; j<_size; j++)
        Moa(i,j) = M(i,j)/a;
    return Moa;
  }
  
  
  template<int _size>
  __host__ __device__ SquareMatrix<true,_size> AssignSymmetric( const SquareMatrix<false,_size> &A) {
    SquareMatrix<true,_size> B;
    for(int i=0; i<_size; i++){
      for(int j=i; j<_size; j++){
        B(i,j) = A(i,j);
      }
    }
    return B;
  } 

  typedef Vector<5> Vector5;
  typedef SquareMatrix<true,5> SymMatrix5x5;
  typedef SquareMatrix<false,5> Matrix5x5;

  __device__ __host__ void tensorProduct(const Vector<5> &u, const Vector<5> &v, SquareMatrix<true,5> &A);
  __device__ __host__ void multiply_S5x5_2x1(const SquareMatrix<true,5> &A, const Vector<2> &B, Vector<5> &V);
  __device__ __host__ void multiply_S5x5_S2x2(const SquareMatrix<true,5> &A, const SquareMatrix<true,2> &B, Vector<10> &V);
  __device__ __host__ void similarity_1x2_S5x5_2x1(const Vector<2> &A, const SquareMatrix<true,5> &B, double &r);
  __device__ __host__ void similarity_5x2_2x2(const Vector<10> &K, const SquareMatrix<true,2> &C, SquareMatrix<true,5> &KCKt);
  __device__ __host__ double similarity_2x1_2x2(const Vector<2> &a, const SquareMatrix<true,2> &C);
  __device__ __host__ SquareMatrix<true,5> similarity_5_5(const SquareMatrix<false,5> &F, const SquareMatrix<true,5> &C);
  __device__ __host__ SquareMatrix<true,5> similarity_5_5_alt(const SquareMatrix<false,5> &F, const SquareMatrix<true,5> &C);
  __device__ __host__ void WeightedAverage(const Vector<5> &x1, const SquareMatrix<true,5> &C1,
                                           const Vector<5> &x2, const SquareMatrix<true,5> &C2,
                                           Vector<5> &x, SquareMatrix<true,5> &C);
  __device__ __host__ Vector<5> operator*(const Vector<10> &M, const Vector<2> &a);
  
}
