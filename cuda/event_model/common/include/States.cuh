#pragma once

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
  bool backward;
};
