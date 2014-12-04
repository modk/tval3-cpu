#ifndef MAT_MULT_H_
#define MAT_MULT_H_

//#include <omp.h>
#include <cmath>
#include <complex>
#include "container.h"
#include "blas_wrap.h"

#include "profile.h"

// overloaded function for matrix-vector multiplication (dense / sparse / dynamically calculated)

template <class Type>
// y = Op(A)*x
// A Row Major, x, y column Major!
inline void gemv(CBLAS_TRANSPOSE TransA, const mat<Type> &A, const mat<Type> &x,
                 mat<Type> &y) {
#ifdef PROFILING
	if (TransA == CblasNoTrans) {
		profile_info[10].valid = true;
		sprintf(profile_info[10].name, "%s (N)\t (%s,%i)\0",
		        __FUNCTION__, __FILE__, __LINE__ - 4);
	} else {
		profile_info[11].valid = true;
		sprintf(profile_info[11].name, "%s (T)\t (%s,%i)\0",
		        __FUNCTION__, __FILE__, __LINE__ - 7);
	}
	double start = get_timestamp();
#endif

	cblas_gemv(CblasRowMajor, TransA, A.dim_y, A.dim_x, (Type)1,
	           A.data(), A.dim_x, x.data(), 1, (Type)0, y.data(), 1);

#ifdef PROFILING
	double stop = get_timestamp();
	if (TransA == CblasNoTrans) {
		profile_info[10].time += (stop - start);
		profile_info[10].runs++;
	} else {
		profile_info[11].time += (stop - start);
		profile_info[11].runs++;
	}
#endif
}

template <class Type>
inline void gemv(CBLAS_TRANSPOSE TransA, const sparse_mat<Type> &A,
                 const mat<Type> &x, mat<Type> &y) {
#ifdef PROFILING
	if ( TransA == CblasNoTrans ) {
		profile_info[10].valid = true;
		sprintf(profile_info[10].name, "%s (N)\t (%s,%i)\0",
		        __FUNCTION__, __FILE__, __LINE__ - 4);
	} else {
		profile_info[11].valid = true;
		sprintf(profile_info[11].name, "%s (T)\t (%s,%i)\0",
		        __FUNCTION__, __FILE__, __LINE__ - 7);
	}
	double start = get_timestamp();
#endif

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

//	mkl_csrmv(&transpose, (int *)&A.dim_y, (int *)&A.dim_x, &alpha, matdescra, (Type *)A.csr_val(), (int *)A.csr_ind(),
//				(int *)A.csr_ptr(), (int *)(A.csr_ptr() + 1), (Type *)x.data(), &beta, y.data());

//	mkl_cscmv(&transpose, (int *)&A.dim_y, (int *)&A.dim_x, &alpha, matdescra, (Type *)A.csc_val(), (int *)A.csc_ind(),
//					(int *)A.csc_ptr(), (int *)(A.csc_ptr() + 1), (Type *)x.data(), &beta, y.data());

	if ( TransA == CblasNoTrans ) {

		mkl_csrmv(&transpose, (int *)&A.dim_y, (int *)&A.dim_x, &alpha,
		          matdescra, (Type *)A.csr_val(), (int *)A.csr_ind(),
		          (int *)A.csr_ptr(),
		          (int *)(A.csr_ptr() + 1),
		          (Type *)x.data(), &beta, y.data());

	} else {

		if ( A.dim_x >= 100 * 100 * 100 )
			mkl_cscmv(&transpose, (int *)&A.dim_y, (int *)&A.dim_x,
			          &alpha, matdescra,
			          (Type *)A.csc_val(), (int *)A.csc_ind(),
			          (int *)A.csc_ptr(),
			          (int *)(A.csc_ptr() + 1),
			          (Type *)x.data(), &beta, y.data());
		else
			mkl_csrmv (&transpose, (int *)&A.dim_y, (int *)&A.dim_x,
			           &alpha, matdescra,
			           (Type *)A.csr_val(), (int *)A.csr_ind(),
			           (int *)A.csr_ptr(),
			           (int *)(A.csr_ptr() + 1),
			           (Type *)x.data(), &beta, y.data());
	}


#ifdef PROFILING
	double stop = get_timestamp();
	if ( TransA == CblasNoTrans ) {
		profile_info[10].time += (stop - start);
		profile_info[10].runs++;
	} else {
		profile_info[11].time += (stop - start);
		profile_info[11].runs++;
	}
#endif
}

// functions for the dynamic calculation of the measurement matrix...
template <class Type>
inline void mult_elementN(CBLAS_TRANSPOSE TransA, int ray, int x_pixel,
                          int y_pixel, int z_pixel, int rv_x, int rv_y,
                          float scale_factor, const Type *x, Type *y) {

	int lin_index = x_pixel + y_pixel * rv_x + z_pixel * rv_x * rv_y;
	y[ray] += x[lin_index] * (Type) scale_factor;

}

template <class Type>
inline void mult_elementT(CBLAS_TRANSPOSE TransA, int ray, int x_pixel,
                          int y_pixel, int z_pixel, int rv_x, int rv_y,
                          float scale_factor, const Type *x, Type *y) {

	int lin_index = x_pixel + y_pixel * rv_x + z_pixel * rv_x * rv_y;
	y[lin_index] += x[ray] * (Type) scale_factor;

}

struct int4 {
	int x;
	int y;
	int z;
	int w;
};

template <class Type>
void gemv(CBLAS_TRANSPOSE TransA, const geometry &A, const mat<Type> &x,
          mat<Type> &y) {
#ifdef PROFILING
	if ( TransA == CblasNoTrans ) {
		profile_info[10].valid = true;
		sprintf(profile_info[10].name, "%s (N)\t (%s,%i)\0",
		        __FUNCTION__, __FILE__, __LINE__ - 3);
	} else {
		profile_info[11].valid = true;
		sprintf(profile_info[11].name, "%s (T)\t (%s,%i)\0",
		        __FUNCTION__, __FILE__, __LINE__ - 3);
	}
	double start = get_timestamp();
#endif

	int len_y =
	        (TransA == CblasNoTrans) ? (A.numMPs * A.numPaths) : (A.rv_x *
	                                                              A.rv_y *
	                                                              A.rv_z);
	memset(y.data(), 0, len_y * sizeof(Type));

	Type *y_loc = y.data();
	const Type *x_loc = x.data();

	int numThreads;
	Type **vecs;

	if ( TransA == CblasNoTrans ) { // nicht transponiert -> Normal

		#pragma omp parallel
		{
			int dx, dy, dz, x_inc, mx, y_inc, my, z_inc, mz,
			    x_pixel, y_pixel, z_pixel, ray, err_1, err_2;

			for (int i = 0; i < A.numMPs; i++) {

				#pragma omp for schedule (dynamic, 1)
				for(int emitter = 0;
				    emitter < A.num_emitters;
				    emitter++) {
					for(int receiver = 0;
					    receiver < A.num_receivers;
					    receiver++) {

						ray = emitter *
						      A.num_receivers +
						      receiver;

						// if current path is not valid (does not satisfy the maxAngle criterion), go on with next path
						if ( A.use_path[ray] == 0 )
							continue;

						ray = i * A.numPaths +
						      A.use_path[ray] - 1;

						// otherwise trace the path using Bresenhams algorithm...
						dx =
						        A.x_receivers[i][
						                receiver] -
						        A.x_emitters[i][emitter];
						dy =
						        A.y_receivers[i][
						                receiver] -
						        A.y_emitters[i][emitter];
						dz =
						        A.z_receivers[i][
						                receiver] -
						        A.z_emitters[i][emitter];

						x_inc = (dx < 0) ? -1 : 1;
						mx = abs(dx);

						y_inc = (dy < 0) ? -1 : 1;
						my = abs(dy);

						z_inc = (dz < 0) ? -1 : 1;
						mz = abs(dz);

						x_pixel =
						        A.x_emitters[i][emitter];
						y_pixel =
						        A.y_emitters[i][emitter];
						z_pixel =
						        A.z_emitters[i][emitter];

						mult_elementN(TransA, ray,
						              x_pixel, y_pixel,
						              z_pixel, A.rv_x,
						              A.rv_y,
						              A.scale_factor,
						              x_loc, y_loc);

						if (mx >= my && mx >= mz) {
							err_1 = 2 * my - mx;
							err_2 = 2 * mz - mx;
							for(int j = 1;
							    j < mx + 1;
							    j++) {
								if (err_1 > 0) {
									y_pixel
									        +=
									                y_inc;
									err_1 -=
									        2
									        *
									        mx;
								}
								if (err_2 > 0) {
									z_pixel
									        +=
									                z_inc;
									err_2 -=
									        2
									        *
									        mx;
								}
								err_1 += 2 * my;
								err_2 += 2 * mz;
								x_pixel +=
								        x_inc;
								mult_elementN(
								        TransA,
								        ray,
								        x_pixel,
								        y_pixel,
								        z_pixel,
								        A.rv_x,
								        A.rv_y,
								        A.scale_factor, x_loc,
								        y_loc);
							}
						} else if(my >= mx && my >=
						          mz) {
							err_1 = 2 * mx - my;
							err_2 = 2 * mz - my;
							for (int j = 1;
							     j < my + 1;
							     j++) {
								if (err_1 > 0) {
									x_pixel
									        +=
									                x_inc;
									err_1 -=
									        2
									        *
									        my;
								}
								if (err_2 > 0) {
									z_pixel
									        +=
									                z_inc;
									err_2 -=
									        2
									        *
									        my;
								}
								err_1 += 2 * mx;
								err_2 += 2 * mz;
								y_pixel +=
								        y_inc;
								mult_elementN(
								        TransA,
								        ray,
								        x_pixel,
								        y_pixel,
								        z_pixel,
								        A.rv_x,
								        A.rv_y,
								        A.scale_factor, x_loc,
								        y_loc);
							}
						} else {
							err_1 = 2 * mx - mz;
							err_2 = 2 * my - mz;
							for (int j = 1;
							     j < mz + 1;
							     j++) {
								if (err_1 > 0) {
									x_pixel
									        +=
									                x_inc;
									err_1 -=
									        2
									        *
									        mz;
								}
								if (err_2 > 0) {
									y_pixel
									        +=
									                y_inc;
									err_2 -=
									        2
									        *
									        mz;
								}
								err_1 += 2 * mx;
								err_2 += 2 * my;
								z_pixel +=
								        z_inc;
								mult_elementN(
								        TransA,
								        ray,
								        x_pixel,
								        y_pixel,
								        z_pixel,
								        A.rv_x,
								        A.rv_y,
								        A.scale_factor, x_loc,
								        y_loc);
							}
						}
					} // end receiver loop
				} // end emitter loop
			} // end MP loop
		} // end OMP parallel

	} else { // transponiert

		#pragma omp parallel
		{
			int dx, dy, dz, x_inc, mx, y_inc, my, z_inc, mz,
			    x_pixel, y_pixel, z_pixel, ray, err_1, err_2;

			#pragma omp single
			{
				numThreads = omp_get_num_threads();
				vecs = new Type* [numThreads];
			}

			int threadIdx = omp_get_thread_num();
			if ( threadIdx == 0 )
				vecs[0] = y_loc;
			else {
				vecs[threadIdx] = new Type[len_y];
				memset(vecs[threadIdx], 0, len_y *
				       sizeof(Type));
			}

			for (int i = 0; i < A.numMPs; i++) {

				#pragma omp for schedule (dynamic, 1)
				for(int emitter = 0;
				    emitter < A.num_emitters;
				    emitter++) {
					for(int receiver = 0;
					    receiver < A.num_receivers;
					    receiver++) {

						ray = emitter *
						      A.num_receivers +
						      receiver;

						// if current path is not valid (does not satisfy the maxAngle criterion), go on with next path
						if ( A.use_path[ray] == 0 )
							continue;

						ray = i * A.numPaths +
						      A.use_path[ray] - 1;

						// otherwise trace the path using Bresenhams algorithm...
						dx =
						        A.x_receivers[i][
						                receiver] -
						        A.x_emitters[i][emitter];
						dy =
						        A.y_receivers[i][
						                receiver] -
						        A.y_emitters[i][emitter];
						dz =
						        A.z_receivers[i][
						                receiver] -
						        A.z_emitters[i][emitter];

						x_inc = (dx < 0) ? -1 : 1;
						mx = abs(dx);

						y_inc = (dy < 0) ? -1 : 1;
						my = abs(dy);

						z_inc = (dz < 0) ? -1 : 1;
						mz = abs(dz);

						x_pixel =
						        A.x_emitters[i][emitter];
						y_pixel =
						        A.y_emitters[i][emitter];
						z_pixel =
						        A.z_emitters[i][emitter];

						mult_elementT(TransA, ray,
						              x_pixel, y_pixel,
						              z_pixel, A.rv_x,
						              A.rv_y,
						              A.scale_factor,
						              x_loc,
						              vecs[threadIdx]);

						if (mx >= my && mx >= mz) {
							err_1 = 2 * my - mx;
							err_2 = 2 * mz - mx;
							for(int j = 1;
							    j < mx + 1;
							    j++) {
								if (err_1 > 0) {
									y_pixel
									        +=
									                y_inc;
									err_1 -=
									        2
									        *
									        mx;
								}
								if (err_2 > 0) {
									z_pixel
									        +=
									                z_inc;
									err_2 -=
									        2
									        *
									        mx;
								}
								err_1 += 2 * my;
								err_2 += 2 * mz;
								x_pixel +=
								        x_inc;
								mult_elementT(
								        TransA,
								        ray,
								        x_pixel,
								        y_pixel,
								        z_pixel,
								        A.rv_x,
								        A.rv_y,
								        A.scale_factor, x_loc,
								        vecs[
								                threadIdx]);
							}
						} else if(my >= mx && my >=
						          mz) {
							err_1 = 2 * mx - my;
							err_2 = 2 * mz - my;
							for (int j = 1;
							     j < my + 1;
							     j++) {
								if (err_1 > 0) {
									x_pixel
									        +=
									                x_inc;
									err_1 -=
									        2
									        *
									        my;
								}
								if (err_2 > 0) {
									z_pixel
									        +=
									                z_inc;
									err_2 -=
									        2
									        *
									        my;
								}
								err_1 += 2 * mx;
								err_2 += 2 * mz;
								y_pixel +=
								        y_inc;
								mult_elementT(
								        TransA,
								        ray,
								        x_pixel,
								        y_pixel,
								        z_pixel,
								        A.rv_x,
								        A.rv_y,
								        A.scale_factor, x_loc,
								        vecs[
								                threadIdx]);
							}
						} else {
							err_1 = 2 * mx - mz;
							err_2 = 2 * my - mz;
							for (int j = 1;
							     j < mz + 1;
							     j++) {
								if (err_1 > 0) {
									x_pixel
									        +=
									                x_inc;
									err_1 -=
									        2
									        *
									        mz;
								}
								if (err_2 > 0) {
									y_pixel
									        +=
									                y_inc;
									err_2 -=
									        2
									        *
									        mz;
								}
								err_1 += 2 * mx;
								err_2 += 2 * my;
								z_pixel +=
								        z_inc;
								mult_elementT(
								        TransA,
								        ray,
								        x_pixel,
								        y_pixel,
								        z_pixel,
								        A.rv_x,
								        A.rv_y,
								        A.scale_factor, x_loc,
								        vecs[
								                threadIdx]);
							}
						}
					} // end receiver loop
				} // end emitter loop
			} // end MP loop
		} // end OMP parallel

		for(int i = 1; i < numThreads; i++) {

			cblas_axpy(len_y, 1, vecs[i], 1, y_loc, 1);

			delete vecs[i];

		}

		delete vecs;

	}

#ifdef PROFILING
	double stop = get_timestamp();
	if ( TransA == CblasNoTrans ) {
		profile_info[10].time += (stop - start);
		profile_info[10].runs++;
	} else {
		profile_info[11].time += (stop - start);
		profile_info[11].runs++;
	}
#endif
}

#endif /* MAT_MULT_H_ */
