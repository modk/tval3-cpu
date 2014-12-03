#ifndef MAT_MULT_H_
#define MAT_MULT_H_

#include <cmath>
#include <complex>
#include "container.h"
#include "blas_wrap.h"

// overloaded function for matrix-vector multiplication (dense / sparse /
// dynamically calculated)

template <class Type>
// y = Op(A)*x
// A Row Major, x, y column Major!
inline void gemv(CBLAS_TRANSPOSE TransA, const mat<Type> &A, const mat<Type> &x,
		mat<Type> &y, int layers) { 
	
	if(layers == 1) { 
		// single layer...
		cblas_gemv(CblasRowMajor, TransA, A.dim_y, A.dim_x, (Type)1, A.data(),
				A.dim_x, x.data(), 1, (Type)0, y.data(), 1);
	} else { 
		// multi layer...
		mat<Type> y_tmp(y.len);

		int N = (TransA == CblasNoTrans) ? A.dim_x : A.dim_y;
		int M = (TransA == CblasNoTrans) ? A.dim_y : A.dim_x;

		cblas_gemm(CblasRowMajor, TransA, CblasTrans, M, layers, N, 1, A.data(),
				A.dim_x, x.data(), N, 1, y_tmp.data(), layers); 
		
		for(int i=0; i < layers; i++) {
			cblas_copy(M, y_tmp.data() + i, layers, y.data() + i*M, 1);
		}
	}
}

template <class Type>
inline void gemv(CBLAS_TRANSPOSE TransA, const sparse_mat<Type> &A, const
		mat<Type> &x, mat<Type> &y, int layers) {

	char transpose;

	if(TransA == CblasTrans)
		transpose = 't';
	else if(TransA == CblasConjTrans)
		transpose = 'c';
	else
		transpose = 'n';

	char *matdescra = (char *)"GLLC";
	Type alpha = 1;
	Type beta = 0;

	if(layers == 1) { 
		// single layer...

		if ( TransA == CblasNoTrans ) {
			mkl_csrmv(&transpose, (int *)&A.dim_y, (int *)&A.dim_x, &alpha, matdescra,
					(Type *)A.csr_val(), (int *)A.csr_ind(), (int *)A.csr_ptr(), 
					(int *)(A.csr_ptr() + 1), (Type *)x.data(), &beta, y.data());
		} else {
			mkl_cscmv(&transpose, (int *)&A.dim_y, (int *)&A.dim_x, &alpha, matdescra,
					(Type *)A.csc_val(), (int *)A.csc_ind(), (int *)A.csc_ptr(), 
					(int *)(A.csc_ptr() + 1), (Type *)x.data(), &beta, y.data());
		}

	} else { 
		// multi layer...

		mat<Type> y_tmp(y.len);
		mat<Type> x_tmp(x.len);

		int N = (TransA == CblasNoTrans) ? A.dim_x : A.dim_y;
		int M = (TransA == CblasNoTrans) ? A.dim_y : A.dim_x;

		for(int i=0; i < layers; i++) {
			cblas_copy(N, x.data() + i*N, 1, x_tmp.data() + i, layers);
		}

		mkl_cscmm(&transpose, (int *)&A.dim_y, &layers, (int *)&A.dim_x, &alpha,
				matdescra, (Type *)A.csc_val(), (int *)A.csc_ind(), (int *)A.csc_ptr(),
				(int *)(A.csc_ptr() + 1), (Type *)x_tmp.data(), &layers, &beta,
				y_tmp.data(), &layers);

		for(int i=0; i < layers; i++) {
			cblas_copy(M, y_tmp.data() + i, layers, y.data() + i*M, 1);
		}

	}
}

// functions for the dynamic calculation of the measurement matrix
template <class Type>
inline void mult_element(CBLAS_TRANSPOSE TransA, int ray, int x_pixel, int
		y_pixel, int ld, float scale_factor, const mat<Type> &x, mat<Type> &y) {

	int lin_index = x_pixel + y_pixel*ld;

	if(TransA == CblasNoTrans) {
		y[ray] += x[lin_index] * (Type) scale_factor;
	} else {
		y[lin_index] += x[ray] * (Type) scale_factor;
	}
}

template <class Type>
// Parameter layers is ignored
void gemv(CBLAS_TRANSPOSE TransA, const geometry &A, const mat<Type> &x, mat<Type> &y, int layers) {

/*
	memset(y.data(), 0, y.len*sizeof(Type));

	mat<Type> **vecs;
	int numthreads;
	#pragma omp parallel
	{
		int dx, dy, x_inc, le, y_inc, m, x_pixel, y_pixel, ray, err_1;

		int threadnum = omp_get_thread_num();

		#pragma omp single
		{
			numthreads = omp_get_num_threads();
			vecs = new mat<Type>* [numthreads];
		}

		if(threadnum==0)
			vecs[0] = &y;
		else
			vecs[threadnum] = new mat<Type>(y.len);


		#pragma omp for //schedule (dynamic, 1)
		for(int emitter=0; emitter < A.num_emitters; emitter++) {
			for(int receiver=0; receiver < A.num_receivers; receiver++) {

				ray = emitter*A.num_receivers + receiver;

				// if current path is not valid (does not satisfy the maxAngle criterion), go on with next path
				if ( A.use_path[ray] == 0 )
					continue;

				// otherwise trace path using Bresenhams algorithm...
				dx = A.x_receivers[receiver] - A.x_emitters[emitter];
				dy = A.y_receivers[receiver] - A.y_emitters[emitter];

				x_inc = (dx < 0) ? -1 : 1;
				le = abs(dx);

				y_inc = (dy < 0) ? -1 : 1;
				m = abs(dy);

				x_pixel = A.x_emitters[emitter];
				y_pixel = A.y_emitters[emitter];


				mult_element(TransA, ray, x_pixel, y_pixel, A.ld, A.scale_factor, x, *vecs[threadnum]);

				if (le >= m) {
					err_1 = 2*m - le;
					for(int j = 1; j < le+1; j++) {
						if (err_1 > 0) {
							y_pixel += y_inc;
							err_1 -= 2*le;
						}
						err_1 += 2*m;
						x_pixel += x_inc;
						mult_element(TransA, ray, x_pixel, y_pixel, A.ld, A.scale_factor, x, *vecs[threadnum]);
					}
				} else {
					err_1 = 2*le - m;
					for (int j = 1; j < m+1; j++) {
						if (err_1 > 0) {
							x_pixel += x_inc;
							err_1 -= 2*m;
						}
						err_1 += 2*le;
						y_pixel += y_inc;
						mult_element(TransA, ray, x_pixel, y_pixel, A.ld, A.scale_factor, x, *vecs[threadnum]);
					}
				}
			}
		}
	}

	for(int i=1; i<numthreads; i++) {
		cblas_axpy(y.len, 1, vecs[i]->data(), 1, y.data(), 1);
		delete vecs[i];
	}

	delete vecs;
*/
}

#endif /* MAT_MULT_H_ */
