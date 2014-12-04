#define PROFILING

#ifdef PROFILING
double t_mult, t_d, t_dt;
bool rec_time;
#endif

#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <complex>
#include <string>
#include <stdio.h>

#include "timestamp.h"
#include "profile.h"

#include "tval3_types.h"
#include "utility.h"
#include "blas_wrap.h"
#include "mat_mult.h"
#include <mkl.h>
#include <fstream>
#include <cfloat>

using namespace std;

//-----------------------------------------------------------------------------
// function headers for external visibility
//-----------------------------------------------------------------------------
#if (1)
extern const tval3_info<float> tval3_cpu_3d(mat<float> &U, const mat<float> &A,
                                            const mat<float> &b,
                                            const tval3_options<float> &opts,
                                            const mat<float> &Ut, int submat_mb,
                                            int virtual_procs_per_core) throw(
        tval3_exception) __attribute__ ((visibility ("default") ));

extern const tval3_info<double> tval3_cpu_3d(mat<float> &U, const mat<float> &A,
                                             const mat<float> &b,
                                             const tval3_options<double> &opts,
                                             const mat<float> &Ut,
                                             int submat_mb,
                                             int virtual_procs_per_core) throw(
        tval3_exception) __attribute__ ((visibility ("default") ));

extern const tval3_info<double> tval3_cpu_3d(mat<double> &U,
                                             const mat<double> &A,
                                             const mat<double> &b,
                                             const tval3_options<double> &opts,
                                             const mat<double> &Ut,
                                             int submat_mb,
                                             int virtual_procs_per_core) throw(
        tval3_exception) __attribute__ ((visibility ("default") ));

extern const tval3_info<float> tval3_cpu_3d(mat<std::complex<float> > &U,
                                            const mat<std::complex<float> > &A,
                                            const mat<std::complex<float> > &b,
                                            const tval3_options<float> &opts,
                                            const mat<std::complex<float> > &Ut,
                                            int submat_mb,
                                            int virtual_procs_per_core)
throw(tval3_exception) __attribute__ ((visibility ("default") ));

extern const tval3_info<double> tval3_cpu_3d(mat<std::complex<double> > &U,
                                             const mat<std::complex<double> > &A,
                                             const mat<std::complex<double> > &b, const tval3_options<double> &opts, const mat<std::complex<double> > &Ut, int submat_mb,
                                             int virtual_procs_per_core) throw(
        tval3_exception) __attribute__ ((visibility ("default") ));

extern const tval3_info<float> tval3_cpu_3d(mat<float> &U,
                                            const sparse_mat<float> &A,
                                            const mat<float> &b,
                                            const tval3_options<float> &opts,
                                            const mat<float> &Ut, int submat_mb,
                                            int virtual_procs_per_core) throw(
        tval3_exception) __attribute__ ((visibility ("default") ));

extern const tval3_info<double> tval3_cpu_3d(mat<float> &U,
                                             const sparse_mat<float> &A,
                                             const mat<float> &b,
                                             const tval3_options<double> &opts,
                                             const mat<float> &Ut,
                                             int submat_mb,
                                             int virtual_procs_per_core) throw(
        tval3_exception) __attribute__ ((visibility ("default") ));

extern const tval3_info<double> tval3_cpu_3d(mat<double> &U,
                                             const sparse_mat<double> &A,
                                             const mat<double> &b,
                                             const tval3_options<double> &opts,
                                             const mat<double> &Ut,
                                             int submat_mb,
                                             int virtual_procs_per_core) throw(
        tval3_exception) __attribute__ ((visibility ("default") ));

extern const tval3_info<float> tval3_cpu_3d(mat<std::complex<float> > &U,
                                            const sparse_mat<std::complex<float> > &A,
                                            const mat<std::complex<float> > &b, const tval3_options<float> &opts, const mat<std::complex<float> > &Ut, int submat_mb,
                                            int virtual_procs_per_core) throw(
        tval3_exception) __attribute__ ((visibility ("default") ));

extern const tval3_info<double> tval3_cpu_3d(mat<std::complex<double> > &U,
                                             const sparse_mat<std::complex<double> > &A,
                                             const mat<std::complex<double> > &b, const tval3_options<double> &opts, const mat<std::complex<double> > &Ut, int submat_mb,
                                             int virtual_procs_per_core) throw(
        tval3_exception) __attribute__ ((visibility ("default") ));

extern const tval3_info<float> tval3_cpu_3d(mat<float> &U, const geometry &A,
                                            const mat<float> &b,
                                            const tval3_options<float> &opts,
                                            const mat<float> &Ut, int submat_mb,
                                            int virtual_procs_per_core) throw(
        tval3_exception) __attribute__ ((visibility ("default") ));

extern const tval3_info<double> tval3_cpu_3d(mat<float> &U, const geometry &A,
                                             const mat<float> &b,
                                             const tval3_options<double> &opts,
                                             const mat<float> &Ut,
                                             int submat_mb,
                                             int virtual_procs_per_core) throw(
        tval3_exception) __attribute__ ((visibility ("default") ));

extern const tval3_info<double> tval3_cpu_3d(mat<double> &U, const geometry &A,
                                             const mat<double> &b,
                                             const tval3_options<double> &opts,
                                             const mat<double> &Ut,
                                             int submat_mb,
                                             int virtual_procs_per_core) throw(
        tval3_exception) __attribute__ ((visibility ("default") ));

extern const tval3_info<float> tval3_cpu_3d(mat<std::complex<float> > &U,
                                            const geometry &A,
                                            const mat<std::complex<float> > &b,
                                            const tval3_options<float> &opts,
                                            const mat<std::complex<float> > &Ut,
                                            int submat_mb,
                                            int virtual_procs_per_core)
throw(tval3_exception) __attribute__ ((visibility ("default") ));

extern const tval3_info<double> tval3_cpu_3d(mat<std::complex<double> > &U,
                                             const geometry &A,
                                             const mat<std::complex<double> > &b,
                                             const tval3_options<double> &opts, const mat<std::complex<double> > &Ut, int submat_mb,
                                             int virtual_procs_per_core)
throw(tval3_exception) __attribute__ ((visibility ("default") ));

#endif

//-----------------------------------------------------------------------------
// tval3_data struct: holds various values, accessed by the functions get_g(),
// update_g(), update_W() and update_mlp()
// the member names correspond to the variable names in the MATLAB-version
//-----------------------------------------------------------------------------
template<class T_comp, class T_scal>
struct tval3_data { // parameters and intermediate results

	// dimensions
	int p; // y-dimension of reconstruction volume
	int q; // x-dimension of reconstruction volume
	int r; // z-dimension of reconstruction volume
	int n; // number of pixels (n = p*q*r)
	int m; // number of measurements

	// penalty parameters
	T_scal mu;
	T_scal beta;
	T_scal muDbeta; // mu/beta

	// multiplier
	mat<T_comp> delta;
	mat<T_comp> sigmaX;
	mat<T_comp> sigmaY;
	mat<T_comp> sigmaZ;

	// lagrangian function and sub terms
	T_comp f;
	T_scal lam1;
	T_scal lam2;
	T_scal lam3;
	T_comp lam4;
	T_comp lam5;

	// matrices / vectors to hold intermediate computational data
	mat<T_comp> Up;
	mat<T_comp> dU; // U-Up, after steepest descent
	mat<T_comp> uup; // U-Up, after "backtracking"
	mat<T_comp> Ux; // gradients in x-direction
	mat<T_comp> Uxp;
	mat<T_comp> dUx;
	mat<T_comp> Uy; // gradients in y-direction
	mat<T_comp> Uyp;
	mat<T_comp> dUy;
	mat<T_comp> Uz; // gradients in y-direction
	mat<T_comp> Uzp;
	mat<T_comp> dUz;
	mat<T_comp> Wx;
	mat<T_comp> Wy;
	mat<T_comp> Wz;
	mat<T_comp> Atb; // A'*b
	mat<T_comp> Au; // A*u
	mat<T_comp> Aup;
	mat<T_comp> dAu;
	mat<T_comp> d; // gradient of objective function (2.28)
	mat<T_comp> g; // sub term of (2.28)
	mat<T_comp> gp;
	mat<T_comp> dg;
	mat<T_comp> g2; // sub term of (2.28)
	mat<T_comp> g2p;
	mat<T_comp> dg2;

	// additional parameters for skipping of multiplication with adjunct matrix
	const unsigned int numInitIterations;
	const T_scal skipMulRatio;
	const T_scal maxRelativeChange;
	unsigned int currInitIteration;
	T_scal currSkipValue;
	mat<T_comp> last_Au;

	tval3_data<T_comp, T_scal>(const mat<T_comp> &U, const mat<T_comp> &b,
	                           const tval3_options<T_scal> &opts) :
	p(U.dim_y), q(U.dim_x), r(U.dim_z), n(U.len), m(b.len), mu(opts.mu0),
	beta(opts.beta0), delta(b.len), 
	sigmaX(U.dim_y, U.dim_x, U.dim_z),
	sigmaY(U.dim_y, U.dim_x, U.dim_z), 
	sigmaZ(U.dim_y, U.dim_x, U.dim_z), 
	Up(U.dim_y, U.dim_x, U.dim_z), 
	dU( U.dim_y, U.dim_x, U.dim_z),
	uup(U.len),
	Ux(U.dim_y, U.dim_x, U.dim_z), Uxp(U.dim_y, U.dim_x, U.dim_z), dUx(
	        U.dim_y, U.dim_x, U.dim_z), Uy(U.dim_y, U.dim_x, U.dim_z),
	Uyp(U.dim_y, U.dim_x, U.dim_z), dUy(U.dim_y, U.dim_x, U.dim_z), Uz(
	        U.dim_y, U.dim_x, U.dim_z), Uzp(U.dim_y, U.dim_x, U.dim_z),
	dUz(U.dim_y, U.dim_x, U.dim_z), Wx(U.dim_y, U.dim_x, U.dim_z), Wy(
	        U.dim_y, U.dim_x, U.dim_z), Wz(U.dim_y, U.dim_x, U.dim_z), Atb(
	        U.len),
	Au(b.len), Aup(b.len), dAu(b.len),
	d(U.dim_y, U.dim_x,
	  U.dim_z), g(U.len), gp(U.len), dg(U.len), g2(U.len), g2p(U.len), dg2(
	        U.len),
	numInitIterations(opts.numInitIterations), skipMulRatio(
	        opts.skipMulRatio), maxRelativeChange(opts.maxRelativeChange),
	currInitIteration(0), currSkipValue(0.0), last_Au(b.len){
	}
};

//-----------------------------------------------------------------------------
// vec_add: z <- alpha*x + y, N: number of elements
//-----------------------------------------------------------------------------
#if (1)
template<class T_mat, class T_scal>
void vec_add(const int N, const int stride, const T_mat *x, const T_scal alpha,
             const T_mat *y, T_mat *z) {
#ifdef PROFILING
	profile_info[0].valid = true;
	sprintf(profile_info[0].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 3);
	double start = get_timestamp();
#endif

	#pragma vector always
	#pragma parallel always
	for(int i = 0; i < N; i++)
		z[i * stride] = x[i * stride] + alpha * y[i * stride];

#ifdef PROFILING
	double stop = get_timestamp();
	profile_info[0].time += (stop - start);
	profile_info[0].runs++;
#endif
}

template<class T_mat, class T_scal>
void vec_add(const int N, const int stride, const std::complex<T_mat> *x,
             const T_scal alpha, const std::complex<T_mat> *y,
             std::complex<T_mat> *z) {

#ifdef PROFILING
	profile_info[0].valid = true;
	sprintf(profile_info[0].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 3);
	double start = get_timestamp();
#endif

	T_mat *data_x = (T_mat *)x;
	T_mat *data_y = (T_mat *)y;
	T_mat *data_z = (T_mat *)z;
	#pragma vector always
	#pragma parallel always
	for(int i = 0; i < N; i++) {
		data_z[2 * i *
		       stride] =
		        data_x[2 * i * stride] + alpha * data_y[2 * i * stride];
		data_z[2 * i * stride +
		       1] =
		        data_x[2 * i * stride +
		               1] + alpha * data_y[2 * i * stride + 1];
	}

#ifdef PROFILING
	double stop = get_timestamp();
	profile_info[0].time += (stop - start);
	profile_info[0].runs++;
#endif
}
#endif

//-----------------------------------------------------------------------------
// scaleb: scales the measurement vector and returns the scale factor.
//-----------------------------------------------------------------------------
template<class T_comp, class T_scal>
T_scal scaleb(mat<T_comp> &b) {
#ifdef PROFILING
	profile_info[1].valid = true;
	sprintf(profile_info[1].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 3);
	double start = get_timestamp();
#endif

	T_scal scl = 1;
	if(b.len > 0) {
		T_scal threshold1 = 0.5;
		T_scal threshold2 = 1.5;
		scl = 1;
		T_comp val;
		T_scal val_abs;
		T_comp bmin = b[0];
		T_comp bmax = bmin;
		T_scal bmin_abs = abs(bmin);
		T_scal bmax_abs = bmin_abs;
		for(int i = 0; i < b.len; i++) {
			val = b[i];
			val_abs = abs(val);
			if(val_abs < bmin_abs) {
				bmin = val;
				bmin_abs = val_abs;
			}
			if(val_abs > bmax_abs) {
				bmax = val;
				bmax_abs = val_abs;
			}
		}
		T_scal b_dif = abs(bmax - bmin);
		if(b_dif < threshold1) {
			scl = threshold1 / b_dif;
			cblas_scal(b.len, scl, b.data(), 1);
		} else if(b_dif > threshold2) {
			scl = threshold2 / b_dif;
			cblas_scal(b.len, scl, b.data(), 1);
		}
	}

#ifdef PROFILING
	double stop = get_timestamp();
	profile_info[1].time += (stop - start);
	profile_info[1].runs++;
#endif

	return scl;
}

//-----------------------------------------------------------------------------
// D
//-----------------------------------------------------------------------------
template<class T_comp, class T_scal>
void D(mat<T_comp> &Ux, mat<T_comp> &Uy, mat<T_comp> &Uz,
       const mat<T_comp> &U) {

	vec_add(U.dim_x * U.dim_y * U.dim_z - 1, 1, U.data() + 1, -1,
	        U.data(), Ux.data());
	vec_add(U.dim_y * U.dim_z, U.dim_x, U.data(), -1,
	        U.data() + U.dim_x - 1, Ux.data() + U.dim_x - 1);

	vec_add((U.dim_y * U.dim_z - 1) * U.dim_x, 1,
	        U.data() + U.dim_x, -1, U.data(), Uy.data());
	for(int z = 0; z < U.dim_z; z++)
		vec_add(U.dim_x, 1,
		        U.data() + z * U.dim_x * U.dim_y, -1, U.data() +
		        (U.dim_y - 1) * U.dim_x + z * U.dim_x * U.dim_y,
		        Uy.data() +
		        (U.dim_y - 1) * U.dim_x + z * U.dim_x * U.dim_y);

	vec_add(U.dim_x * U.dim_y * (U.dim_z - 1), 1,
	        U.data() + U.dim_x * U.dim_y, -1, U.data(), Uz.data());
	vec_add(U.dim_x * U.dim_y, 1, U.data(), -1,
	        U.data() + (U.dim_z - 1) * U.dim_x * U.dim_y,
	        Uz.data() + (U.dim_z - 1) * U.dim_x * U.dim_y);

}

//-----------------------------------------------------------------------------
// Dt
//-----------------------------------------------------------------------------
template<class T_comp, class T_scal>
void Dt(mat<T_comp> &res, const mat<T_comp> &X, const mat<T_comp> &Y,
        const mat<T_comp> &Z) {

	vec_add(X.len - 1, 1, X.data(), -1, X.data() + 1, res.data() + 1);
	vec_add(X.dim_y * X.dim_z, X.dim_x, X.data() + X.dim_x - 1, -1,
	        X.data(), res.data());

	cblas_axpy((Y.dim_y * Y.dim_z - 1) * Y.dim_x, (T_comp)1,
	           Y.data(), 1, res.data() + Y.dim_x, 1);
	cblas_axpy((Y.dim_y * Y.dim_z - 1) * Y.dim_x, (T_comp) - 1,
	           Y.data() + Y.dim_x, 1, res.data() + Y.dim_x, 1);
	for(int z = 0; z < Y.dim_z; z++) {
		cblas_axpy(Y.dim_x, (T_comp)1,
		           Y.data() + (Y.dim_y - 1) * Y.dim_x + z * Y.dim_x * Y.dim_y, 1,
		           res.data() + z * Y.dim_x * Y.dim_y, 1);
		cblas_axpy(Y.dim_x, (T_comp) - 1,
		           Y.data() + z * Y.dim_x * Y.dim_y, 1,
		           res.data() + z * Y.dim_x * Y.dim_y, 1);
	}

	cblas_axpy(Z.dim_y * Z.dim_x * (Z.dim_z - 1), (T_comp)1,
	           Z.data(), 1, res.data() + Z.dim_y * Z.dim_x, 1);
	cblas_axpy(Z.dim_y * Z.dim_x * (Z.dim_z - 1), (T_comp) - 1,
	           Z.data() + Z.dim_y * Z.dim_x, 1,
	           res.data() + Z.dim_y * Z.dim_x, 1);
	cblas_axpy(Z.dim_y * Z.dim_x, (T_comp)1,
	           Z.data() + (Z.dim_z - 1) * Z.dim_x * Z.dim_y, 1,
	           res.data(), 1);
	cblas_axpy(Z.dim_y * Z.dim_x, (T_comp) - 1, Z.data(), 1, res.data(), 1);

}

//-----------------------------------------------------------------------------
// get_g_mult: calculates: Au = A * U and g = A' * Au
//-----------------------------------------------------------------------------
#if (1)

// standard version (parameter submat_mb not used)
template<class T_comp, class T_scal, class T_A>
void get_g_mult(tval3_data<T_comp, T_scal> &data, const T_A &A,
                const mat<T_comp> &U, int submat_mb, bool force_mul = true) {

	gemv(CblasNoTrans, A, U, data.Au);

	if ( data.skipMulRatio == 0 ) {

		gemv(CblasConjTrans, A, data.Au, data.g);

	} else {

		mat<T_comp> deltaAu = data.Au;
		cblas_axpy(deltaAu.len, (T_comp) - 1,
		           data.last_Au.data(), 1, deltaAu.data(), 1);
		T_scal nrm_deltaAu = cblas_nrm2(deltaAu.len, deltaAu.data(), 1);
		T_scal nrm_Au = cblas_nrm2(data.Au.len, data.Au.data(), 1);
		T_scal relChange = nrm_deltaAu / nrm_Au;

		if ( force_mul == true ) {
			// do always, if forced (used for initialization outside of loop and
			// if deepest descent fails

			data.last_Au = data.Au;
			gemv(CblasConjTrans, A, data.Au, data.g);
			data.currSkipValue = 0;
			//cout << "force\n";

		} else if ( data.currInitIteration < data.numInitIterations ) {
			// do also for specified number of first iterations, after that do only, if:

			data.last_Au = data.Au;
			gemv(CblasConjTrans, A, data.Au, data.g);
			data.currInitIteration++;
			data.currSkipValue = 0;
			//cout << "init " << data.currInitIteration << "\n";

		} else if ( data.currSkipValue < 1 ) {
			// specified skip-ratio is reached or if

			data.last_Au = data.Au;
			gemv(CblasConjTrans, A, data.Au, data.g);
			data.currSkipValue += data.skipMulRatio;
			//cout << "skip " << data.currSkipValue << "\n";

		} else if ( relChange > data.maxRelativeChange) {
			// data.Au changed to much

			data.last_Au = data.Au;
			gemv(CblasConjTrans, A, data.Au, data.g);
			//cout << "maxChange " << relChange << "\n";

		} else {

			data.currSkipValue -= 1;
			//cout << "..\n";

		}
	}

}

// special version for dense matrices (cache blocking for single layer
// reconstructions)
// submat_mb: size of submatrices in MB
template<class T_comp, class T_scal>
void get_g_mult(tval3_data<T_comp, T_scal> &data, const mat<T_comp> &A,
                const mat<T_comp> &U, int submat_mb, bool force_mul = true) {

	int submat_elements = submat_mb * 1024 * 1024 / sizeof(T_comp);

	mat<T_comp> g_tmp(data.n);

	int submat_dim_y = max(submat_elements / data.n, 1);

	int n_y = (data.m + submat_dim_y - 1) / submat_dim_y;

	int r;
	memset(data.g.data(), 0, data.n * sizeof(T_comp));
	for(int y = 0; y < n_y; y++) {
		r = min(submat_dim_y, data.m - y * submat_dim_y);

		cblas_gemv(CblasRowMajor, CblasNoTrans, r, data.n, 1,
		           A.data() + y * submat_dim_y * data.n, data.n,
		           U.data(), 1, 0, data.Au.data() + y * submat_dim_y,
		           1);

		cblas_gemv(CblasRowMajor, CblasConjTrans, r, data.n, 1,
		           A.data() + y * submat_dim_y * data.n, data.n,
		           data.Au.data() + y * submat_dim_y, 1, 0,
		           g_tmp.data(),
		           1);

		cblas_axpy(data.n, 1, g_tmp.data(), 1, data.g.data(), 1);
	}

}
#endif

//-----------------------------------------------------------------------------------------------------------------------
// get_g
//-----------------------------------------------------------------------------------------------------------------------
template<class T_comp, class T_scal, class T_A>
void get_g(tval3_data<T_comp, T_scal> &data, const T_A &A, const mat<T_comp> &U,
           const mat<T_comp> &b, int submat_mb, bool force_mul = true) {

	get_g_mult(data, A, U, submat_mb, force_mul);

	cblas_axpy(data.n, (T_comp) - 1, data.Atb.data(), 1, data.g.data(), 1);

	// update g2, lam2, lam4
	mat<T_comp> Vx(data.p, data.q, data.r);
	mat<T_comp> Vy(data.p, data.q, data.r);
	mat<T_comp> Vz(data.p, data.q, data.r);
	vec_add(data.n, 1, data.Ux.data(), (T_scal) - 1,
	        data.Wx.data(), Vx.data());                                        // Vx = Ux - Wx
	vec_add(data.n, 1, data.Uy.data(), (T_scal) - 1,
	        data.Wy.data(), Vy.data());                                        // Vy = Uy - Wy
	vec_add(data.n, 1, data.Uz.data(), (T_scal) - 1,
	        data.Wz.data(), Vz.data());                                        // Vz = Uz - Wz
	Dt<T_comp, T_scal>(data.g2, Vx, Vy, Vz);

	data.lam2 = real(cblas_dotc(data.n, Vx.data(), 1, Vx.data(), 1));
	data.lam2 += real(cblas_dotc(data.n, Vy.data(), 1, Vy.data(), 1));
	data.lam2 += real(cblas_dotc(data.n, Vz.data(), 1, Vz.data(), 1));
	data.lam4 =
	        real(cblas_dotc(data.n, data.sigmaX.data(), 1, Vx.data(), 1));
	data.lam4 += real(cblas_dotc(data.n, data.sigmaY.data(), 1,
	                             Vy.data(), 1));
	data.lam4 += real(cblas_dotc(data.n, data.sigmaZ.data(), 1,
	                             Vz.data(), 1));

	//update lam3, lam5
	mat<T_comp> Aub(data.m);
	vec_add(data.m, 1, data.Au.data(), (T_scal) - 1, b.data(), Aub.data()); // Aub = Au - b
	data.lam3 = real(cblas_dotc(data.m, Aub.data(), 1, Aub.data(), 1));
	data.lam5 =
	        real(cblas_dotc(data.m, data.delta.data(), 1, Aub.data(), 1));

	data.f =
	        (data.lam1 + data.beta / 2 * data.lam2 + data.mu / 2 *
	         data.lam3) - data.lam4 - data.lam5;

}

//-----------------------------------------------------------------------------------------------------------------------
// get_tau
//-----------------------------------------------------------------------------------------------------------------------
template<class T_comp, class T_scal, class T_A>
T_scal get_tau(tval3_data<T_comp, T_scal> &data, const T_A &A, bool fst_iter) {

	T_scal tau;

	// calculate tau
	if(fst_iter) {
		mat<T_comp> dx(data.p, data.q, data.r);
		mat<T_comp> dy(data.p, data.q, data.r);
		mat<T_comp> dz(data.p, data.q, data.r);
		D<T_comp, T_scal>(dx, dy, dz, data.d);
		T_comp dDd = cblas_dotc(data.n, dx.data(), 1, dx.data(), 1) +
		             cblas_dotc(data.n, dy.data(), 1, dy.data(), 1) +
		             cblas_dotc(data.n, dz.data(), 1, dz.data(), 1);
		T_comp dd = cblas_dotc(data.n, data.d.data(), 1,
		                       data.d.data(), 1);
		mat<T_comp> Ad(data.m);
		// Ad=A*d
		gemv(CblasNoTrans, A, data.d, Ad);
		T_comp Add = cblas_dotc(data.m, Ad.data(), 1, Ad.data(), 1);

		tau = abs(dd / (dDd + data.muDbeta * Add));
	} else {
		vec_add(data.n, 1, data.g.data(), -1,
		        data.gp.data(), data.dg.data());
		vec_add(data.n, 1, data.g2.data(), -1,
		        data.g2p.data(), data.dg2.data());
		T_comp ss = cblas_dotc(data.n,
		                       data.uup.data(), 1, data.uup.data(), 1);
		mat<T_comp> tmp(data.dg2);
		cblas_axpy(data.n, data.muDbeta, data.dg.data(), 1,
		           tmp.data(), 1);
		T_comp sy =
		        cblas_dotc(data.n, data.uup.data(), 1, tmp.data(), 1);
		tau = abs(ss / sy);
	}

	return tau;
}

//-----------------------------------------------------------------------------------------------------------------------
// nonneg
//-----------------------------------------------------------------------------------------------------------------------
#if (1)
template<class T_mat>
void nonneg(mat<T_mat> &U) {
#ifdef PROFILING
	profile_info[2].valid = true;
	sprintf(profile_info[2].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 3);
	double start = get_timestamp();
#endif

	T_mat val;
	#pragma parallel always
	#pragma vector always
	for(int i = 0; i < U.len; i++) {
		val = U[i];
		U[i] = (val < 0) ? 0 : val;
	}

#ifdef PROFILING
	double stop = get_timestamp();
	profile_info[2].time += (stop - start);
	profile_info[2].runs++;
#endif
}

template<class T_mat>
void nonneg(mat<complex<T_mat> > &U) {
#ifdef PROFILING
	profile_info[2].valid = true;
	sprintf(profile_info[2].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 3);
	double start = get_timestamp();
#endif

	T_mat val;
	T_mat *data = (T_mat *) U.data();
	#pragma parallel always
	#pragma vector always
	for(int i = 0; i < U.len; i++) {
		val = data[2 * i];
		data[2 * i] = (val < 0) ? 0 : val;
		data[2 * i + 1] = 0;
	}

#ifdef PROFILING
	double stop = get_timestamp();
	profile_info[2].time += (stop - start);
	profile_info[2].runs++;
#endif
}
#endif

//-----------------------------------------------------------------------------------------------------------------------
// descend
//-----------------------------------------------------------------------------------------------------------------------
template<class T_comp, class T_scal, class T_A>
void descend(mat<T_comp> &U, tval3_data<T_comp, T_scal> &data, const T_A &A,
             const mat<T_comp> &b, T_scal tau, bool non_neg,
             int submat_mb, bool force_mul = true) {

	cblas_axpy(data.n, -1 * tau, data.d.data(), 1, U.data(), 1); // U = U - tau*d
	if(non_neg) {
		nonneg(U);
	}
	D<T_comp, T_scal>(data.Ux, data.Uy, data.Uz, U);
	get_g<T_comp, T_scal>(data, A, U, b, submat_mb, force_mul);

}

//-----------------------------------------------------------------------------------------------------------------------
// update_g
//-----------------------------------------------------------------------------------------------------------------------
template<class T_comp, class T_scal, class T_A>
void update_g(tval3_data<T_comp, T_scal> &data, const T_scal alpha,
              const T_A &A, mat<T_comp> &U, const mat<T_comp> &b) {

	vec_add(data.n, 1, data.gp.data(), alpha, data.dg.data(),
	        data.g.data());
	vec_add(data.n, 1, data.g2p.data(), alpha,
	        data.dg2.data(), data.g2.data());
	vec_add(data.n, 1, data.Up.data(), alpha, data.dU.data(), U.data());
	vec_add(data.m, 1, data.Aup.data(), alpha,
	        data.dAu.data(), data.Au.data());
	vec_add(data.n, 1, data.Uxp.data(), alpha,
	        data.dUx.data(), data.Ux.data());
	vec_add(data.n, 1, data.Uyp.data(), alpha,
	        data.dUy.data(), data.Uy.data());
	vec_add(data.n, 1, data.Uzp.data(), alpha,
	        data.dUz.data(), data.Uz.data());

	// update lam2, lam4
	mat<T_comp> Vx(data.p, data.q, data.r);
	mat<T_comp> Vy(data.p, data.q, data.r);
	mat<T_comp> Vz(data.p, data.q, data.r);
	vec_add(data.n, 1, data.Ux.data(), (T_scal) - 1,
	        data.Wx.data(), Vx.data());                      // Vx = Ux - Wx
	vec_add(data.n, 1, data.Uy.data(), (T_scal) - 1,
	        data.Wy.data(), Vy.data());                      // Vy = Uy - Wy
	vec_add(data.n, 1, data.Uz.data(), (T_scal) - 1,
	        data.Wz.data(), Vz.data());                      // Vz = Uz - Wz

	data.lam2 = real(cblas_dotc(data.n, Vx.data(), 1, Vx.data(), 1));
	data.lam2 += real(cblas_dotc(data.n, Vy.data(), 1, Vy.data(), 1));
	data.lam2 += real(cblas_dotc(data.n, Vz.data(), 1, Vz.data(), 1));
	data.lam4 =
	        real(cblas_dotc(data.n, data.sigmaX.data(), 1, Vx.data(), 1));
	data.lam4 += real(cblas_dotc(data.n, data.sigmaY.data(), 1,
	                             Vy.data(), 1));
	data.lam4 += real(cblas_dotc(data.n, data.sigmaZ.data(), 1,
	                             Vz.data(), 1));

	// update lam3, lam5
	mat<T_comp> Aub(data.m);
	vec_add(data.m, 1, data.Au.data(), (T_scal) - 1, b.data(), Aub.data()); // Aub = Au - b
	data.lam3 = real(cblas_dotc(data.m, Aub.data(), 1, Aub.data(), 1));
	data.lam5 =
	        real(cblas_dotc(data.m, data.delta.data(), 1, Aub.data(), 1));

	data.f = data.lam1 + data.beta / 2 * data.lam2 + data.mu / 2 *
	         data.lam3 - data.lam4 - data.lam5;

}

//-----------------------------------------------------------------------------------------------------------------------
// min_u
//-----------------------------------------------------------------------------------------------------------------------
template<class T_comp, class T_scal, class T_A>
void min_u(tval3_data<T_comp, T_scal> &data, mat<T_comp> &U, T_scal &gam,
           const T_A &A, const mat<T_comp> &b,
           const tval3_options<T_scal> &opts, T_comp C, bool fst_iter,
           int submat_mb) {

	T_scal tau, alpha, c_armij;

	tau = get_tau<T_comp, T_scal, T_A>(data, A, fst_iter);

	// keep previous values
	data.Up = U; data.gp = data.g; data.g2p = data.g2;
	data.Aup = data.Au; data.Uxp = data.Ux; data.Uyp = data.Uy;
	data.Uzp = data.Uz;

	// one step steepest descent
	descend<T_comp, T_scal, T_A>(U, data, A, b, tau, opts.nonneg, submat_mb,
	                             false);

	// NMLA
	alpha = 1;
	vec_add(data.n, 1, U.data(), -1, data.Up.data(), data.dU.data()); // Ud = U - Up

	// c_armij = d'*d*tau*c*beta
	c_armij = real(cblas_dotc(data.n, data.d.data(), 1,
	                          data.d.data(), 1) * tau * opts.c * data.beta);

	if(abs(data.f) > abs(C - alpha * c_armij)) {  // Armijo condition

		vec_add(data.n, 1, data.g.data(), -1,
		        data.gp.data(), data.dg.data());                // dg=g-gp
		vec_add(data.n, 1, data.g2.data(), -1,
		        data.g2p.data(), data.dg2.data());              // dg2=g2-g2p
		vec_add(data.m, 1, data.Au.data(), -1,
		        data.Aup.data(), data.dAu.data());              // dAu=Au-Aup
		vec_add(data.n, 1, data.Ux.data(), -1,
		        data.Uxp.data(), data.dUx.data());              // dUx=Ux-Uxp
		vec_add(data.n, 1, data.Uy.data(), -1,
		        data.Uyp.data(), data.dUy.data());              // Uy = Uy-Uyp
		vec_add(data.n, 1, data.Uz.data(), -1,
		        data.Uzp.data(), data.dUz.data());              // Uz = Uz-Uzp

		int cnt = 0;

		while(abs(data.f) > abs(C - alpha * c_armij)) { // Armijo condition

			if(cnt == 5) { // "backtracking" not successful
				gam *= opts.rate_gam;
				tau =
				        get_tau<T_comp, T_scal, T_A>(data, A,
				                                     true);
				U = data.Up;
				descend<T_comp, T_scal, T_A>(U, data, A, b, tau,
				                             opts.nonneg,
				                             submat_mb, true);
				break;
			}
			alpha *= opts.gamma;

			update_g<T_comp, T_scal, T_A>(data, alpha, A, U, b);

			cnt++;
		}
	}
}

//-----------------------------------------------------------------------------
// get_gradient
//-----------------------------------------------------------------------------
template<class T_comp, class T_scal>
void get_gradient(tval3_data<T_comp, T_scal> &data, const mat<T_comp> &DtsAtd) {

	// d = g2 + muDbeta*g + DtsAtd  (DtsAtd has opposite sign as the MATLAB-version!)
	data.d = DtsAtd;
	cblas_axpy(data.n, 1, data.g2.data(), 1, data.d.data(), 1);
	cblas_axpy(data.n, data.muDbeta, data.g.data(), 1, data.d.data(), 1);

}

//-----------------------------------------------------------------------------
// shrinkage
//-----------------------------------------------------------------------------
#if (1)

template<class T_mat, class T_scal>
void shrinkage(tval3_data<T_mat, T_scal> &data) {
#ifdef PROFILING
	profile_info[3].valid = true;
	sprintf(profile_info[3].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 3);
	double start = get_timestamp();
#endif

	T_scal sum = 0;
	T_scal temp_wx, temp_wy, temp_wz;
	T_mat Uxbar, Uybar, Uzbar;
	#pragma parallel always
	#pragma vector always
	for(int i = 0; i < data.n; i++) {
		Uxbar = data.Ux[i] - data.sigmaX[i] / data.beta;
		Uybar = data.Uy[i] - data.sigmaY[i] / data.beta;
		Uzbar = data.Uz[i] - data.sigmaZ[i] / data.beta;

		temp_wx = abs(Uxbar) - 1 / data.beta;
		temp_wy = abs(Uybar) - 1 / data.beta;
		temp_wz = abs(Uzbar) - 1 / data.beta;
		temp_wx = (temp_wx >= 0) ? temp_wx : 0;
		temp_wy = (temp_wy >= 0) ? temp_wy : 0;
		temp_wz = (temp_wz >= 0) ? temp_wz : 0;
		sum += temp_wx + temp_wy + temp_wz;
		data.Wx[i] = (Uxbar >= 0) ? temp_wx : -1 * temp_wx;
		data.Wy[i] = (Uybar >= 0) ? temp_wy : -1 * temp_wy;
		data.Wz[i] = (Uzbar >= 0) ? temp_wz : -1 * temp_wz;
	}
	data.lam1 = sum;

#ifdef PROFILING
	double stop = get_timestamp();
	profile_info[3].time += (stop - start);
	profile_info[3].runs++;
#endif
}

template<class T_mat, class T_scal>
void shrinkage(tval3_data<complex<T_mat>, T_scal> &data) {

#ifdef PROFILING
	profile_info[3].valid = true;
	sprintf(profile_info[3].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 3);
	double start = get_timestamp();
#endif

	T_mat sum = 0;
	T_mat temp_wx, temp_wy, temp_wz;
	T_mat Uxbar_real, Uxbar_imag, Uxbar_abs, Uybar_real, Uybar_imag,
	      Uybar_abs, Uzbar_real, Uzbar_imag, Uzbar_abs;
	T_mat *Ux_data = (T_mat *) data.Ux.data();
	T_mat *Uy_data = (T_mat *) data.Uy.data();
	T_mat *Uz_data = (T_mat *) data.Uz.data();
	T_mat *sigmaX_data = (T_mat *) data.sigmaX.data();
	T_mat *sigmaY_data = (T_mat *) data.sigmaY.data();
	T_mat *sigmaZ_data = (T_mat *) data.sigmaZ.data();
	T_mat *Wx_data = (T_mat *) data.Wx.data();
	T_mat *Wy_data = (T_mat *) data.Wy.data();
	T_mat *Wz_data = (T_mat *) data.Wz.data();
	const int len = data.n;

	#pragma parallel always
	#pragma vector always
	for(int i = 0; i < len; i++) {
		Uxbar_real = Ux_data[2 * i] - sigmaX_data[2 * i] / data.beta;
		Uxbar_imag =
		        Ux_data[2 * i + 1] - sigmaX_data[2 * i + 1] / data.beta;
		Uybar_real = Uy_data[2 * i] - sigmaY_data[2 * i] / data.beta;
		Uybar_imag =
		        Uy_data[2 * i + 1] - sigmaY_data[2 * i + 1] / data.beta;
		Uzbar_real = Uz_data[2 * i] - sigmaZ_data[2 * i] / data.beta;
		Uzbar_imag =
		        Uz_data[2 * i + 1] - sigmaZ_data[2 * i + 1] / data.beta;

		Uxbar_abs = sqrt(
		        Uxbar_real * Uxbar_real + Uxbar_imag * Uxbar_imag);
		Uybar_abs = sqrt(
		        Uybar_real * Uybar_real + Uybar_imag * Uybar_imag);
		Uzbar_abs = sqrt(
		        Uzbar_real * Uzbar_real + Uzbar_imag * Uzbar_imag);

		temp_wx = Uxbar_abs - 1 / data.beta;
		temp_wy = Uybar_abs - 1 / data.beta;
		temp_wz = Uzbar_abs - 1 / data.beta;
		temp_wx = (temp_wx >= 0) ? temp_wx : 0;
		temp_wy = (temp_wy >= 0) ? temp_wy : 0;
		temp_wz = (temp_wz >= 0) ? temp_wz : 0;
		sum += temp_wx + temp_wy + temp_wz;

		Wx_data[2 * i] =
		        (Uxbar_abs > 0) ? temp_wx * Uxbar_real / Uxbar_abs : 0;
		Wx_data[2 * i + 1] =
		        (Uxbar_abs > 0) ? temp_wx * Uxbar_imag / Uxbar_abs : 0;
		Wy_data[2 * i] =
		        (Uybar_abs > 0) ? temp_wy * Uybar_real / Uybar_abs : 0;
		Wy_data[2 * i + 1] =
		        (Uybar_abs > 0) ? temp_wy * Uybar_imag / Uybar_abs : 0;
		Wz_data[2 * i] =
		        (Uzbar_abs > 0) ? temp_wz * Uybar_real / Uzbar_abs : 0;
		Wz_data[2 * i + 1] =
		        (Uzbar_abs > 0) ? temp_wz * Uybar_imag / Uzbar_abs : 0;
	}
	data.lam1 = sum;

#ifdef PROFILING
	double stop = get_timestamp();
	profile_info[3].time += (stop - start);
	profile_info[3].runs++;
#endif
}
#endif

//-----------------------------------------------------------------------------
// update_W
//-----------------------------------------------------------------------------
template<class T_comp, class T_scal>
void update_W(tval3_data<T_comp, T_scal> &data) {

	data.f -= (data.lam1 + data.beta / 2 * data.lam2 - data.lam4);

	shrinkage(data);

	// update g2, lam2, lam4
	mat<T_comp> Vx(data.p, data.q, data.r);
	mat<T_comp> Vy(data.p, data.q, data.r);
	mat<T_comp> Vz(data.p, data.q, data.r);
	vec_add(data.n, 1, data.Ux.data(), (T_scal) - 1,
	        data.Wx.data(), Vx.data());                     // Vx = Ux - Wx
	vec_add(data.n, 1, data.Uy.data(), (T_scal) - 1,
	        data.Wy.data(), Vy.data());                     // Vy = Uy - Wy
	vec_add(data.n, 1, data.Uz.data(), (T_scal) - 1,
	        data.Wz.data(), Vz.data());                     // Vz = Uz - Wz
	Dt<T_comp, T_scal>(data.g2, Vx, Vy, Vz);

	data.lam2 = real(cblas_dotc(data.n, Vx.data(), 1, Vx.data(), 1));
	data.lam2 += real(cblas_dotc(data.n, Vy.data(), 1, Vy.data(), 1));
	data.lam2 += real(cblas_dotc(data.n, Vz.data(), 1, Vz.data(), 1));
	data.lam4 =
	        real(cblas_dotc(data.n, data.sigmaX.data(), 1, Vx.data(), 1));
	data.lam4 += real(cblas_dotc(data.n, data.sigmaY.data(), 1,
	                             Vy.data(), 1));
	data.lam4 += real(cblas_dotc(data.n, data.sigmaZ.data(), 1,
	                             Vz.data(), 1));

	data.f += (data.lam1 + data.beta / 2 * data.lam2 - data.lam4);

}

//-----------------------------------------------------------------------------
// update_mlp
//-----------------------------------------------------------------------------
template<class T_comp, class T_scal>
void update_mlp(tval3_data<T_comp, T_scal> &data, const mat<T_comp> &b) {

	data.f += (data.lam4 + data.lam5);

	mat<T_comp> Vx(data.p, data.q, data.r);
	mat<T_comp> Vy(data.p, data.q, data.r);
	mat<T_comp> Vz(data.p, data.q, data.r);
	vec_add(data.n, 1, data.Ux.data(), (T_scal) - 1,
	        data.Wx.data(), Vx.data());                     // Vx = Ux - Wx
	vec_add(data.n, 1, data.Uy.data(), (T_scal) - 1,
	        data.Wy.data(), Vy.data());                     // Vy = Uy - Wy
	vec_add(data.n, 1, data.Uz.data(), (T_scal) - 1,
	        data.Wz.data(), Vz.data());                     // Vz = Uz - Wz

	cblas_axpy(data.n, -1 * data.beta, Vx.data(), 1, data.sigmaX.data(), 1); // sigmaX -= beta*Vx
	cblas_axpy(data.n, -1 * data.beta, Vy.data(), 1, data.sigmaY.data(), 1); // sigmaY -= beta*Vy
	cblas_axpy(data.n, -1 * data.beta, Vz.data(), 1, data.sigmaZ.data(), 1); // sigmaz -= beta*Vz

	data.lam4 =
	        real(cblas_dotc(data.n, data.sigmaX.data(), 1, Vx.data(), 1));
	data.lam4 += real(cblas_dotc(data.n, data.sigmaY.data(), 1,
	                             Vy.data(), 1));
	data.lam4 += real(cblas_dotc(data.n, data.sigmaZ.data(), 1,
	                             Vz.data(), 1));

	mat<T_comp> Aub(data.m);
	vec_add(data.m, 1, data.Au.data(), (T_scal) - 1, b.data(), Aub.data()); // Aub = Au - b
	cblas_axpy(data.m, -1 * data.mu, Aub.data(), 1, data.delta.data(), 1); // delta -= mu*Aub
	data.lam5 =
	        real(cblas_dotc(data.m, data.delta.data(), 1, Aub.data(), 1));

	data.f -= (data.lam4 + data.lam5);

}

//-----------------------------------------------------------------------------
// check_params
//-----------------------------------------------------------------------------
#if (1)
template<class T_comp>
void check_params(mat<T_comp> &U, const mat<T_comp> &A, const mat<T_comp> &b,
                  const mat<T_comp> &Ut) throw(tval3_exception) {
#ifdef PROFILING
	profile_info[4].valid = true;
	sprintf(profile_info[4].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 3);
	double start = get_timestamp();
#endif

	if (U.format != mat_row_major || Ut.format != mat_row_major ||
	    A.format != mat_row_major)
		throw tval3_exception(
		              string(
		                      "Argument error: Matrices must be in row major format!"));


	else if (Ut.len > 0 && Ut.len != U.len)
		throw tval3_exception(
		              string(
		                      "Argument error: U and Ut must have the same size!"));


	else if (U.len != A.dim_x)
		throw tval3_exception(
		              string(
		                      "Argument error: the length of U must be equal to A.dim_x!"));


	else if(b.len != A.dim_y)
		throw tval3_exception(
		              string(
		                      "Argument error: b.len must be equal to A.dim_y!"));



#ifdef PROFILING
	double stop = get_timestamp();
	profile_info[4].time += (stop - start);
	profile_info[4].runs++;
#endif
}

template<class T_comp>
void check_params(mat<T_comp> &U,
                  const sparse_mat<T_comp> &A,
                  const mat<T_comp> &b,
                  const mat<T_comp> &Ut) throw(
        tval3_exception) {
#ifdef PROFILING
	profile_info[4].valid = true;
	sprintf(profile_info[4].name,
	        "%s\t (%s,%i)\0", __FUNCTION__,
	        __FILE__,
	        __LINE__ - 3);
	double start = get_timestamp();
#endif

	if(U.format != mat_col_major ||
	   Ut.format != mat_col_major)
		throw tval3_exception(
		              string(
		                      "Argument error: Matrices must be in row major format!"));


	if(A.format != sparse_mat_both)
		throw tval3_exception(
		              string(
		                      "Argument error: A must be available in csr and csc format!"));


	if(Ut.len > 0 && Ut.len != U.len)
		throw tval3_exception(
		              string(
		                      "Argument error: U and Ut must have the same size!"));


	else if(U.len != A.dim_x)
		throw tval3_exception(
		              string(
		                      "Argument error: the length of U must be equal to A.dim_x!"));


	else if(b.len != A.dim_y)
		throw tval3_exception(
		              string(
		                      "Argument error: b.len must be equal to A.dim_y!"));



#ifdef PROFILING
	double stop = get_timestamp();
	profile_info[4].time += (stop - start);
	profile_info[4].runs++;
#endif
}

template<class T_comp>
void check_params(mat<T_comp> &U,
                  const geometry &A,
                  const mat<T_comp> &b,
                  const mat<T_comp> &Ut) throw(
        tval3_exception) {
#ifdef PROFILING
	profile_info[4].valid = true;
	sprintf(profile_info[4].name,
	        "%s\t (%s,%i)\0", __FUNCTION__,
	        __FILE__,
	        __LINE__ - 3);
	double start = get_timestamp();
#endif

	if(U.format != mat_row_major ||
	   Ut.format != mat_row_major)
		throw tval3_exception(
		              string(
		                      "Argument error: Matrices must be in row major format!"));


	else if(Ut.len > 0 && Ut.len != U.len)
		throw tval3_exception(
		              string(
		                      "Argument error: U and Ut must have the same size!"));


	else if(b.len != A.numPaths * A.numMPs)
		throw tval3_exception(
		              string(
		                      "Argument error: b.len must be equal to A.num_emitters * A.num_receivers!"));



#ifdef PROFILING
	double stop = get_timestamp();
	profile_info[4].time += (stop - start);
	profile_info[4].runs++;
#endif
}
#endif


//-----------------------------------------------------------------------------
// tval3_cpu_3d: main function
//-----------------------------------------------------------------------------
template<class T_comp, class T_scal, class T_A>
const tval3_info<T_scal> tval3_cpu_3d(mat<T_comp> &U, const T_A &A,
                                      const mat<T_comp> &b,
                                      const tval3_options<T_scal> &opts,
                                      const mat<T_comp> &Ut,
                                      int submat_mb = 5,
                                      int virtual_procs_per_core = 2)
throw(tval3_exception) {

	double start0, stop0;

#ifdef PROFILING
	double start_loop, stop_loop;
	t_mult = 0; t_d = 0; t_dt = 0; rec_time = false;
#endif

	tval3_info<T_scal> info;
	start0 = get_timestamp();
	try {

		check_params (U, A, b, Ut);

		omp_set_num_threads(omp_get_num_procs() /
		                    virtual_procs_per_core);

		mat<T_comp> bs = b; // create a copy of b (scaling)
		tval3_data<T_comp, T_scal> data(U, bs, opts);
		T_scal muf = opts.mu;
		T_scal betaf = opts.beta, beta0 = 0;
		mat<T_comp> DtsAtd(data.n);
		T_scal gam = opts.gam;
		T_scal scl = scaleb<T_comp, T_scal>(bs);
		gemv(CblasConjTrans, A, bs, data.Atb); // Atb = A'*bs
		U = data.Atb; // initial guess: U=A'*b
		if(data.mu > muf) data.mu = muf;
		if(data.beta > betaf) data.beta = betaf;
		data.muDbeta = data.mu / data.beta;
		mat <T_comp> rcdU(U), UrcdU(data.n);
		D<T_comp, T_scal>(data.Ux, data.Uy, data.Uz, U);
		shrinkage(data);
		get_g<T_comp, T_scal>(data, A, U, bs, submat_mb, true);
		get_gradient<T_comp, T_scal>(data, DtsAtd);
		T_scal Q = 1, Qp;
		T_comp C = data.f;
		int count_outer = 0, count_total = 0;
		bool fst_iter;
		T_scal RelChg, RelChgOut = 0, nrmup = 0;

#ifdef PROFILING
		rec_time = true;
		start_loop = get_timestamp();
#endif

		while((count_outer < opts.maxcnt) &&
		      (count_total < opts.maxit)) {

			cout << "count_outer = " << count_outer << "\n";

			fst_iter = true;
			while(count_total < opts.maxit) {

				// u subproblem
				min_u<T_comp, T_scal>(data, U, gam, A, bs, opts,
				                      C, fst_iter, submat_mb);

				// shrinkage like step
				update_W<T_comp, T_scal>(data);

				// update reference values
				Qp = Q, Q = gam * Qp + 1; C =
				        (gam * Qp * C + data.f) / Q;
				vec_add(data.n, 1, U.data(), -1,
				        data.Up.data(), data.uup.data());

				// compute gradient
				get_gradient<T_comp, T_scal>(data, DtsAtd);

				// begin: calculate relative error
				/*mat<T_comp> Uscal = U;
				   for(int i=0; i < data.n; i++)
				        Uscal[i] = Uscal[i]/scl;

				   mat<T_comp> deltaU = Ut;
				   cblas_axpy(Ut.len, (T_comp)-1, Uscal.data(), 1, deltaU.data(), 1);
				   T_scal nrm_deltaU = cblas_nrm2(deltaU.len, deltaU.data(), 1);
				   T_scal nrm_Ut = cblas_nrm2(Ut.len, Ut.data(), 1);
				   T_scal rel_error = nrm_deltaU/nrm_Ut;
				   cout << "count_total = " << count_total << " " <<
				                 rel_error * 100 << "%\n";*/
				// end: calculate relative error

				count_total++;

				// calculate relative change
				nrmup = cblas_nrm2(data.n, data.Up.data(), 1);
				RelChg =
				        cblas_nrm2(data.n, data.uup.data(),
				                   1) / nrmup;

				if((RelChg < opts.tol_inn) && (!fst_iter))
					break;

				fst_iter = false;

			}
			count_outer++;

			// calculate relative change
			vec_add(data.n, 1, U.data(), -1,
			        rcdU.data(), UrcdU.data());                // UrcdU = U - rcdU
			RelChgOut = cblas_nrm2(data.n, UrcdU.data(), 1) / nrmup;
			rcdU = U;

			if(RelChgOut < opts.tol)
				break;

			// update multipliers
			update_mlp<T_comp, T_scal>(data, bs);

			// update penalty parameters for continuation scheme
			beta0 = data.beta;
			data.beta *= opts.rate_cnt;
			data.mu *= opts.rate_cnt;
			if(data.beta > betaf) data.beta = betaf;
			if(data.mu > muf) data.mu = muf;
			data.muDbeta = data.mu / data.beta;

			// update f
			data.f = data.lam1 + data.beta / 2 * data.lam2 +
			         data.mu / 2 * data.lam3 - data.lam4 -
			         data.lam5;

			// DtsAtd = beta0/beta*d
			DtsAtd = data.d;
			cblas_scal(data.n, beta0 / data.beta, DtsAtd.data(), 1);

			// compute gradient
			get_gradient<T_comp, T_scal>(data, DtsAtd);

			// reset reference values
			gam = opts.gam; Q = 1; C = data.f;
		}

		// scale U. Take real part only, if opts.isreal==true
		for(int i = 0; i < data.n; i++) {
			if(opts.isreal)
				U[i] = (T_comp)real(U[i]);
			U[i] = U[i] / scl;
		}

		stop0 = get_timestamp();

#ifdef PROFILING
		stop_loop = get_timestamp();
		rec_time = false;

		ofstream file;
		file.open("profile.txt");
		file << "loop [ms]: " <<
		        (stop_loop - start_loop) * 1000 << "\n";
		file << "mult [ms]: " << t_mult * 1000 << "\n";
		file << "d [ms]: " << t_d * 1000 << "\n";
		file << "dt [ms]: " << t_dt * 1000 << "\n";
		file.close();

		for (int i = 0; i < 100; i++ ) {

			if (profile_info[i].valid == true  )
				// print if more then 1% of total runtime
				printf(
				        "%s:\t %.2f ms (runs: %i, total: %.1f ms)\n",
				        profile_info[i].name,
				        profile_info[i].time * 1000 /
				        profile_info[i].runs,
				        profile_info[i].runs,
				        profile_info[i].time * 1000);

		}

#endif

		// generate return value
		info.secs = stop0 - start0;
		info.outer_iters = count_outer;
		info.total_iters = count_total;
		info.rel_chg = RelChgOut;
		if(Ut.len == U.len) {
			mat<T_comp> deltaU = Ut;
			cblas_axpy(Ut.len, (T_comp) - 1,
			           U.data(), 1, deltaU.data(), 1);
			T_scal nrm_deltaU = cblas_nrm2(deltaU.len,
			                               deltaU.data(), 1);
			T_scal nrm_Ut = cblas_nrm2(Ut.len, Ut.data(), 1);
			info.rel_error = nrm_deltaU / nrm_Ut;
		} else {
			info.rel_error = 0;
		}
	} catch (tval3_exception) {
		throw;
	} catch (const std::exception &ex) {
		string msg = "Internal error: ";
		msg += ex.what();
		throw tval3_exception(msg);
	}

	return info;
}

//-----------------------------------------------------------------------------
// tval3_cpu_3d: specialized callable functions
//-----------------------------------------------------------------------------
#if (1)
const tval3_info<float> tval3_cpu_3d(mat<float> &U, const mat<float> &A,
                                     const mat<float> &b,
                                     const tval3_options<float> &opts,
                                     const mat<float> &Ut, int submat_mb = 5,
                                     int virtual_procs_per_core =
                                             2) throw(tval3_exception)
{
	return tval3_cpu_3d<float, float, mat<float> >(U, A, b, opts, Ut,
	                                               submat_mb,
	                                               virtual_procs_per_core);
}

const tval3_info<double> tval3_cpu_3d(mat<float> &U, const mat<float> &A,
                                      const mat<float> &b,
                                      const tval3_options<double> &opts,
                                      const mat<float> &Ut, int submat_mb = 5,
                                      int virtual_procs_per_core =
                                              2) throw(tval3_exception)
{
	return tval3_cpu_3d<float, double, mat<float> >(U, A, b, opts, Ut,
	                                                submat_mb,
	                                                virtual_procs_per_core);
}

const tval3_info<double> tval3_cpu_3d(mat<double> &U, const mat<double> &A,
                                      const mat<double> &b,
                                      const tval3_options<double> &opts,
                                      const mat<double> &Ut, int submat_mb = 5,
                                      int virtual_procs_per_core =
                                              2) throw(tval3_exception)
{

	return tval3_cpu_3d<double, double, mat<double> >(U, A, b, opts, Ut,
	                                                  submat_mb,
	                                                  virtual_procs_per_core);
}

const tval3_info<float> tval3_cpu_3d(mat<complex<float> > &U,
                                     const mat<complex<float> > &A,
                                     const mat<complex<float> > &b,
                                     const tval3_options<float> &opts,
                                     const mat<complex<float> > &Ut,
                                     int submat_mb = 5,
                                     int virtual_procs_per_core =
                                             2) throw(tval3_exception)
{
	return tval3_cpu_3d<std::complex<float>, float,
	                    mat<std::complex<float> > >(U, A, b, opts, Ut,
	                                                submat_mb,
	                                                virtual_procs_per_core);
}

const tval3_info<double> tval3_cpu_3d(mat<complex<double> > &U,
                                      const mat<complex<double> > &A,
                                      const mat<complex<double> > &b,
                                      const tval3_options<double> &opts,
                                      const mat<complex<double> > &Ut,
                                      int submat_mb = 5,
                                      int virtual_procs_per_core =
                                              2) throw(tval3_exception)
{
	return tval3_cpu_3d<std::complex<double>, double,
	                    mat<std::complex<double> > >(U, A, b, opts, Ut,
	                                                 submat_mb,
	                                                 virtual_procs_per_core);
}

const tval3_info<float> tval3_cpu_3d(mat<float> &U, const sparse_mat<float> &A,
                                     const mat<float> &b,
                                     const tval3_options<float> &opts,
                                     const mat<float> &Ut, int submat_mb = 5,
                                     int virtual_procs_per_core =
                                             2) throw(tval3_exception)
{
	return tval3_cpu_3d<float, float, sparse_mat<float> >(U, A, b, opts, Ut,
	                                                      submat_mb,
	                                                      virtual_procs_per_core);
}

const tval3_info<double> tval3_cpu_3d(mat<float> &U, const sparse_mat<float> &A,
                                      const mat<float> &b,
                                      const tval3_options<double> &opts,
                                      const mat<float> &Ut, int submat_mb = 5,
                                      int virtual_procs_per_core =
                                              2) throw(tval3_exception)
{
	return tval3_cpu_3d<float, double, sparse_mat<float> >(U, A, b, opts,
	                                                       Ut, submat_mb,
	                                                       virtual_procs_per_core);
}

const tval3_info<double> tval3_cpu_3d(mat<double> &U,
                                      const sparse_mat<double> &A,
                                      const mat<double> &b,
                                      const tval3_options<double> &opts,
                                      const mat<double> &Ut, int submat_mb = 5,
                                      int virtual_procs_per_core =
                                              2) throw(tval3_exception)
{
	return tval3_cpu_3d<double, double, sparse_mat<double> >(U, A, b, opts,
	                                                         Ut, submat_mb,
	                                                         virtual_procs_per_core);
}

const tval3_info<float> tval3_cpu_3d(mat<complex<float> > &U,
                                     const sparse_mat<complex<float> > &A,
                                     const mat<complex<float> > &b,
                                     const tval3_options<float> &opts,
                                     const mat<complex<float> > &Ut,
                                     int submat_mb = 5,
                                     int virtual_procs_per_core =
                                             2) throw(tval3_exception)
{
	return tval3_cpu_3d<std::complex<float>, float,
	                    sparse_mat<std::complex<float> > >(U, A, b, opts,
	                                                       Ut, submat_mb,
	                                                       virtual_procs_per_core);
}

const tval3_info<double> tval3_cpu_3d(mat<complex<double> > &U,
                                      const sparse_mat<complex<double> > &A,
                                      const mat<complex<double> > &b,
                                      const tval3_options<double> &opts,
                                      const mat<complex<double> > &Ut,
                                      int submat_mb = 5,
                                      int virtual_procs_per_core =
                                              2) throw(tval3_exception)
{
	return tval3_cpu_3d<std::complex<double>, double,
	                    sparse_mat<std::complex<double> > >(U, A, b, opts,
	                                                        Ut, submat_mb,
	                                                        virtual_procs_per_core);
}

const tval3_info<float> tval3_cpu_3d(mat<float> &U, const geometry &A,
                                     const mat<float> &b,
                                     const tval3_options<float> &opts,
                                     const mat<float> &Ut, int submat_mb = 5,
                                     int virtual_procs_per_core =
                                             2) throw(tval3_exception)
{
	return tval3_cpu_3d<float, float, geometry>(U, A, b, opts, Ut,
	                                            submat_mb,
	                                            virtual_procs_per_core);
}

const tval3_info<double> tval3_cpu_3d(mat<float> &U, const geometry &A,
                                      const mat<float> &b,
                                      const tval3_options<double> &opts,
                                      const mat<float> &Ut, int submat_mb = 5,
                                      int virtual_procs_per_core =
                                              2) throw(tval3_exception)
{
	return tval3_cpu_3d<float, double, geometry>(U, A, b, opts, Ut,
	                                             submat_mb,
	                                             virtual_procs_per_core);
}

const tval3_info<double> tval3_cpu_3d(mat<double> &U, const geometry &A,
                                      const mat<double> &b,
                                      const tval3_options<double> &opts,
                                      const mat<double> &Ut, int submat_mb = 5,
                                      int virtual_procs_per_core =
                                              2) throw(tval3_exception)
{
	return tval3_cpu_3d<double, double, geometry>(U, A, b, opts, Ut,
	                                              submat_mb,
	                                              virtual_procs_per_core);
}

const tval3_info<float> tval3_cpu_3d(mat<complex<float> > &U, const geometry &A,
                                     const mat<complex<float> > &b,
                                     const tval3_options<float> &opts,
                                     const mat<complex<float> > &Ut,
                                     int submat_mb = 5,
                                     int virtual_procs_per_core =
                                             2) throw(tval3_exception)
{
	return tval3_cpu_3d<std::complex<float>, float, geometry>(U, A, b, opts,
	                                                          Ut, submat_mb,
	                                                          virtual_procs_per_core);
}

const tval3_info<double> tval3_cpu_3d(mat<complex<double> > &U,
                                      const geometry &A,
                                      const mat<complex<double> > &b,
                                      const tval3_options<double> &opts,
                                      const mat<complex<double> > &Ut,
                                      int submat_mb = 5,
                                      int virtual_procs_per_core =
                                              2) throw(tval3_exception)
{
	return tval3_cpu_3d<std::complex<double>, double, geometry>(U, A, b,
	                                                            opts, Ut,
	                                                            submat_mb,
	                                                            virtual_procs_per_core);
}
#endif
