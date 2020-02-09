#ifndef NTK_H
#define NTK_H

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <pybind11/pybind11.h>
#include <vector>

// Matrix class which can be moved into python with zero copy.
// Matrix data is store in row major format.
// Matrix elements are zero indexed.
template <class T>
class Matrix {
public:
    // Ctor, zero initialized
    Matrix(const int nrow, const int ncol) :
        m_nrow(nrow), m_ncol(ncol) {
      m_data = new T[nrow * ncol];
      for (int i = 0; i < nrow * ncol; ++i) m_data[i] = 0.0;
    }

    // Copy constructor
    Matrix(const Matrix& m) :
        m_nrow(m.nrow()), m_ncol(m.ncol()) {
      m_data = new T[m_nrow * m_ncol];
      memcpy(m_data, m.const_data(), m_nrow * m_ncol * sizeof(T));
    }

    // Destructor
    ~Matrix() { delete m_data; }

    // Getters
    T* data() { return m_data; }
    const T* const_data() const { return m_data; }
    const T  value(int i, int j) const { return m_data[m_ncol * i + j]; }
    const int nrow() const { return m_nrow; }
    const int ncol() const { return m_ncol; }

    // Setters
    T& operator()(const int i, const int j) {
        // This method is not exposed to python and is only used in C++.
        // In C++ we ensure that i, j will not be out of bound.
        // Thus we skip boundary check in this method. 
        return m_data[m_ncol * i + j];
    }

    void set_val(const int i, const int j, const T val) {
        // This method is exposed to python and thus needs boundary check.
        if (i < 0 || i >= m_nrow || j < 0 || j >= m_ncol) {
            throw std::runtime_error("index (" + std::to_string(i) + \
                ", " + std::to_string(j) + ") is out of bound");
        }
        m_data[m_ncol*i + j] = val;
    }

    void set_zero() {
      for (int i = 0; i < m_nrow * m_ncol; ++i) m_data[i] = 0.0;
    }

    void copy(const Matrix& m) {
        if (m_nrow != m.nrow() || m_ncol != m.ncol()) {
            throw std::runtime_error("copy matrix shape doesn't match this matrix");
        }
        memcpy(m_data, m.const_data(), m_nrow * m_ncol * sizeof(T));
    }

    // print the matrix for debugging purpose
    void print() {
        for (int i = 0; i < m_nrow * m_ncol; ++i) {
            std::cout << m_data[i] << ",";
            if (i % m_ncol == 0) std::cout << std::endl;
        }
    }

private:
    const int m_nrow;
    const int m_ncol;
    T* m_data;
};

// This function compute and update the S and H matrix
//   given two datasets with n1 data points and n2 data points.
// p1: the sqrt of L2 norm of data set 1, shape: (n1,)
// p2: the sqrt of L2 norm of data set 2, shape: (n2,)
// The python wrapper guarantees that p1 and p2 > 1e-9 to avoid divison by 0.
// S: the activation covariance matrix at dep, shape: (n1, n2)
// H: is the Neural Tangent Kernel matrix at dep, shape: (N1, N2)
// Reference paper: https://arxiv.org/pdf/1910.01663.pdf
// Reference python implementation: https://github.com/LeoYu/neural-tangent-kernel-UCI/blob/master/NTK.py
// Comparing with python implementation, this C++ implementation
//   compute everything in place and thus significantly saves memory.
template <class T>
void Ntk(const std::vector<T>& p1,
         const std::vector<T>& p2,
         Matrix<T>& S,
         Matrix<T>& H) {
    // The following logic implements the recursive relation
    //   of covariance matrix S and neural kernel matrix H in place.
    // The recursive relation is described in page 3 of
    //   https://arxiv.org/pdf/1905.12173.pdf, where:
    // H_k = S_k + H_(k-1) * kappa0(Sn)
    for (int i = 0; i < S.nrow(); ++i) { 
        for (int j = 0; j < S.ncol(); ++j) {
            // Sn is correlation between data i and j,
            // and thus must be in [-1, 1]
            const T Sn = std::min(std::max(S(i, j) / (p1[i] * p2[j]), -1.0f), 1.0f);
            // Update activation covariance matrix S
            const T kappa1 = (Sn * (M_PI - std::acos(Sn)) + std::sqrt(1.0 - Sn * Sn)) / M_PI;
            S(i, j) = kappa1 * p1[i] * p2[j];
            // Update kernel matrix H if indicated by update_H
            const T kappa0 = (M_PI - std::acos(Sn)) / M_PI;
            H(i, j) = S(i, j) + H(i, j) * kappa0;
        }
    }
} 

#endif // NTK_H
