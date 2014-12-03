#ifndef BLAS_WRAP_H_
#define BLAS_WRAP_H_

#include <complex>
#include <mkl_cblas.h>
#include <mkl_spblas.h>

// overloaded wrapper functions for blas / mkl sparse blas calls...
inline void cblas_scal(int N, float alpha, float *X, int incX) {
	cblas_sscal(N, alpha, X, incX);
}

inline void cblas_scal(int N, double alpha, double *X, int incX) {
	cblas_dscal(N, alpha, X, incX);
}

inline void cblas_scal(int N, std::complex<float> alpha, std::complex<float> *X,
		int incX) { 
	cblas_cscal(N, &alpha, X, incX);
}

inline void cblas_scal(int N, std::complex<double> alpha, std::complex<double>
		*X, int incX) { 
	cblas_zscal(N, &alpha, X, incX);
}

inline void cblas_scal(int N, float alpha, std::complex<float> *X, int incX) {
	cblas_csscal(N, alpha, X, incX);
}

inline void cblas_scal(int N, double alpha, std::complex<double> *X, int incX) {
	cblas_zdscal(N, alpha, X, incX);
}

inline void cblas_axpy(int N, float alpha, const float *X, int incX, float *Y,
		int incY) { 
	cblas_saxpy(N, alpha, X, incX, Y, incY);
}

inline void cblas_axpy(int N, double alpha, const double *X, int incX, double
		*Y, int incY) { 
	cblas_daxpy(N, alpha, X, incX, Y, incY);
}

inline void cblas_axpy(int N, std::complex<float> alpha, const
		std::complex<float> *X, int incX, std::complex<float> *Y, int incY) {
	cblas_caxpy(N, &alpha, X, incX, Y, incY);
}

inline void cblas_axpy(int N, std::complex<double> alpha, const
		std::complex<double> *X, int incX, std::complex<double> *Y, int incY) {
	cblas_zaxpy(N, &alpha, X, incX, Y, incY);
}


inline void cblas_copy(int N, const float *X, int incX, float *Y, int incY) {
	cblas_scopy(N, X, incX, Y, incY);
}

inline void cblas_copy(int N, const double *X, int incX, double *Y, int incY) {
	cblas_dcopy(N, X, incX, Y, incY);
}

inline void cblas_copy(int N, const std::complex<float> *X, int incX,
		std::complex<float> *Y, int incY) { 
	cblas_ccopy(N,X, incX, Y, incY);
}

inline void cblas_copy(int N, const std::complex<double> *X, int incX,
		std::complex<double> *Y, int incY) { 
	cblas_zcopy(N, X, incX, Y, incY);
}

inline float cblas_dotc(int N, const float *X, int incX, const float *Y, 
		int incY) { 
	return cblas_sdot(N, X, incX, Y, incY);
}

inline double cblas_dotc(int N, const double *X, int incX, const double *Y, 
		int incY) { 
	return cblas_ddot(N, X, incX, Y, incY);
}

inline std::complex<float> cblas_dotc(int N, const std::complex<float> *X, 
		int incX, const std::complex<float> *Y, int incY) { 
	
	std::complex<float> result;
	cblas_cdotc_sub(N, X, incX, Y, incY, &result);
	return result;
}

inline std::complex<double> cblas_dotc(int N, const std::complex<double> *X, 
		int incX, const std::complex<double> *Y, int incY) { 
	
	std::complex<double> result;
	cblas_zdotc_sub(N, X, incX, Y, incY, &result);
	return result;
}

inline float cblas_nrm2(int N, const float *X, int incX) {
	return cblas_snrm2(N, X, incX);
}

inline double cblas_nrm2(int N, const double *X, int incX) {
	return cblas_dnrm2(N, X, incX);
}

inline float cblas_nrm2(int N, const std::complex<float> *X, int incX) {
	return cblas_scnrm2(N, X, incX);
}

inline double cblas_nrm2(int N, const std::complex<double> *X, int incX) {
	return cblas_dznrm2(N, X, incX);
}

inline void cblas_gemv(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, int M, int N,
		float alpha, const float *A, int lda, const float *X, int incX, float beta,
		float *Y, int incY) {

	cblas_sgemv(Order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

inline void cblas_gemv(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, int M, int N,
		double alpha, const double *A, int lda, const double *X, int incX, double
		beta, double *Y, int incY) {
	
	cblas_dgemv(Order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

inline void cblas_gemv(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, int M, int N,
		std::complex<float> alpha, const std::complex<float> *A, int lda, 
		const std::complex<float> *X, int incX,
		std::complex<float> beta, std::complex<float> *Y, int incY) {

	cblas_cgemv(Order, TransA, M, N, &alpha, A, lda, X, incX, &beta, Y, incY);
}

inline void cblas_gemv(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, int M, int N,
		std::complex<double> alpha, const std::complex<double> *A, int lda, const
		std::complex<double> *X, int incX,
		std::complex<double> beta, std::complex<double> *Y, int incY) {

	cblas_zgemv(Order, TransA, M, N, &alpha, A, lda, X, incX, &beta, Y, incY);
}

inline void cblas_gemm(const  CBLAS_ORDER Order, const  CBLAS_TRANSPOSE TransA,
                 const  CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc) {

	cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void cblas_gemm(const  CBLAS_ORDER Order, const  CBLAS_TRANSPOSE TransA,
                 const  CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc) {

	cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void cblas_gemm(const  CBLAS_ORDER Order, const  CBLAS_TRANSPOSE TransA,
                 const  CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const std::complex<float> alpha, const std::complex<float> *A,
                 const int lda, const std::complex<float> *B, const int ldb,
                 const std::complex<float> beta, std::complex<float> *C, const int ldc) {

	cblas_cgemm(Order, TransA, TransB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

inline void cblas_gemm(const  CBLAS_ORDER Order, const  CBLAS_TRANSPOSE TransA,
                 const  CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const std::complex<double> alpha, const std::complex<double> *A,
                 const int lda, const std::complex<double> *B, const int ldb,
                 const std::complex<double> beta, std::complex<double> *C, const int ldc) {

	cblas_zgemm(Order, TransA, TransB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

inline void mkl_cscmv(char *transa, MKL_INT *m, MKL_INT *k, float *alpha, char
		*matdescra, float  *val, MKL_INT *indx, MKL_INT *pntrb, MKL_INT *pntre,
		float *x, float *beta, float *y) {

	mkl_scscmv(transa, m, k, alpha, matdescra, val, indx,  pntrb, pntre, x, beta, y);
}

inline void mkl_cscmv(char *transa, MKL_INT *m, MKL_INT *k, double *alpha, char
		*matdescra, double  *val, MKL_INT *indx, MKL_INT *pntrb, MKL_INT *pntre,
		double *x, double *beta, double *y) {

	mkl_dcscmv(transa, m, k, alpha, matdescra, val, indx,  pntrb, pntre, x, beta, y);
}

inline void mkl_cscmv(char *transa, MKL_INT *m, MKL_INT *k, 
		std::complex<float> *alpha, char *matdescra, std::complex<float>  *val, 
		MKL_INT *indx,  MKL_INT *pntrb, MKL_INT *pntre, std::complex<float> *x,
		std::complex<float> *beta, std::complex<float> *y) {

	mkl_ccscmv(transa, m, k, (MKL_Complex8 *)alpha, matdescra, (MKL_Complex8 *)val, indx,  pntrb, pntre,
			(MKL_Complex8 *)x, (MKL_Complex8 *)beta, (MKL_Complex8 *)y);
}

inline void mkl_cscmv(char *transa, MKL_INT *m, MKL_INT *k, 
		std::complex<double> *alpha, char *matdescra, std::complex<double>  *val, 
		MKL_INT *indx,  MKL_INT *pntrb, MKL_INT *pntre, std::complex<double> *x,
		std::complex<double> *beta, std::complex<double> *y) {

	mkl_zcscmv(transa, m, k, (MKL_Complex16 *)alpha, matdescra, (MKL_Complex16 *)val, indx,  pntrb, pntre,
			(MKL_Complex16 *)x, (MKL_Complex16 *)beta, (MKL_Complex16 *)y);
}

void mkl_cscmm(char *transa, MKL_INT *m, MKL_INT *n, MKL_INT *k, float *alpha,
		char *matdescra, float  *val, MKL_INT *indx,  MKL_INT *pntrb, 
		MKL_INT *pntre, float *b, MKL_INT *ldb, float *beta, float *c, MKL_INT *ldc) {

	mkl_scscmm(transa, m, n, k, alpha, matdescra, val, indx,  pntrb, pntre, b, ldb, beta, c, ldc);
}

void mkl_cscmm(char *transa, MKL_INT *m, MKL_INT *n, MKL_INT *k, double *alpha,
		char *matdescra, double *val, MKL_INT *indx,  MKL_INT *pntrb, MKL_INT
		*pntre, double *b, MKL_INT *ldb, double *beta, double *c, MKL_INT *ldc) {

	mkl_dcscmm(transa, m, n, k, alpha, matdescra, val, indx,  pntrb, pntre, b, ldb, beta, c, ldc);
}

void mkl_cscmm(char *transa, MKL_INT *m, MKL_INT *n, MKL_INT *k,
		std::complex<float> *alpha, char *matdescra, std::complex<float>  *val,
		MKL_INT *indx,  MKL_INT *pntrb, MKL_INT *pntre, std::complex<float> *b,
		MKL_INT *ldb, std::complex<float> *beta, std::complex<float> *c, 
		MKL_INT *ldc) {

	mkl_ccscmm(transa, m, n, k, (MKL_Complex8 *)alpha, matdescra, (MKL_Complex8 *)val, indx,  pntrb, pntre,
			(MKL_Complex8 *)b, ldb, (MKL_Complex8 *)beta, (MKL_Complex8 *)c, ldc);
}

void mkl_cscmm(char *transa, MKL_INT *m, MKL_INT *n, MKL_INT *k,
		std::complex<double> *alpha, char *matdescra, std::complex<double>  *val,
		MKL_INT *indx,  MKL_INT *pntrb, MKL_INT *pntre, std::complex<double> *b,
		MKL_INT *ldb, std::complex<double> *beta, std::complex<double> *c, 
		MKL_INT *ldc) {

	mkl_zcscmm(transa, m, n, k, (MKL_Complex16 *)alpha, matdescra, 
			(MKL_Complex16 *)val, indx,  pntrb, pntre, (MKL_Complex16 *)b, ldb, 
			(MKL_Complex16 *)beta, (MKL_Complex16 *)c, ldc);
}

inline void mkl_csrmv(char *transa, MKL_INT *m, MKL_INT *k, float *alpha, char
		*matdescra, float  *val, MKL_INT *indx, MKL_INT *pntrb, MKL_INT *pntre,
		float *x, float *beta, float *y) {

	mkl_scsrmv(transa, m, k, alpha, matdescra, val, indx,  pntrb, pntre, x, beta, y);
}

inline void mkl_csrmv(char *transa, MKL_INT *m, MKL_INT *k, double *alpha, char
		*matdescra, double  *val, MKL_INT *indx, MKL_INT *pntrb, MKL_INT *pntre,
		double *x, double *beta, double *y) {

	mkl_dcsrmv(transa, m, k, alpha, matdescra, val, indx,  pntrb, pntre, x, beta, y);
}

inline void mkl_csrmv(char *transa, MKL_INT *m, MKL_INT *k, std::complex<float>
		*alpha, char *matdescra, std::complex<float>  *val, MKL_INT *indx,
		MKL_INT *pntrb, MKL_INT *pntre, std::complex<float> *x, 
		std::complex<float> *beta, std::complex<float> *y) {

	mkl_ccsrmv(transa, m, k, (MKL_Complex8 *)alpha, matdescra, 
			(MKL_Complex8 *)val, indx,  pntrb, pntre, (MKL_Complex8 *)x, 
			(MKL_Complex8 *)beta, (MKL_Complex8 *)y);
}

inline void mkl_csrmv(char *transa, MKL_INT *m, MKL_INT *k, 
		std::complex<double> *alpha, char *matdescra, std::complex<double>  *val, 
		MKL_INT *indx,  MKL_INT *pntrb, MKL_INT *pntre, std::complex<double> *x, 
		std::complex<double> *beta, std::complex<double> *y) {

	mkl_zcsrmv(transa, m, k, (MKL_Complex16 *)alpha, matdescra, 
			(MKL_Complex16 *)val, indx,  pntrb, pntre, (MKL_Complex16 *)x, 
			(MKL_Complex16 *)beta, (MKL_Complex16 *)y);
}

#endif /* BLAS_WRAP_H_ */
