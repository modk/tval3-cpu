#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <complex>
#include <string>
#include "tval3_types.h"
#include "utility.h"
#include "timestamp.h"
#include "blas_wrap.h"
#include "mat_mult.h"
#include <mkl.h>
#include "stdio.h"

using namespace std;

extern const tval3_info<float> tval3_cpu_2d(mat<float> &U, const mat<float> &A,
		const mat<float> &b, const tval3_options<float> &opts, const mat<float> &Ut,
		int submat_mb, int virtual_procs_per_core) 
	throw(tval3_exception) __attribute__ ((visibility ("default") ));

extern const tval3_info<double> tval3_cpu_2d(mat<float> &U, const mat<float> &A,
		const mat<float> &b, const tval3_options<double> &opts, const mat<float>
		&Ut, int submat_mb, int virtual_procs_per_core) 
	throw(tval3_exception) __attribute__ ((visibility ("default") ));

extern const tval3_info<double> tval3_cpu_2d(mat<double> &U, const mat<double>
		&A, const mat<double> &b, const tval3_options<double> &opts, const
		mat<double> &Ut, int submat_mb, int virtual_procs_per_core)
	throw(tval3_exception)  __attribute__ ((visibility ("default") ));

extern const tval3_info<float> tval3_cpu_2d(mat<std::complex<float> > &U, const
		mat<std::complex<float> > &A, const mat<std::complex<float> > &b, const
		tval3_options<float> &opts,  const mat<std::complex<float> > &Ut, int
		submat_mb, int virtual_procs_per_core)
	throw(tval3_exception)  __attribute__ ((visibility ("default") ));

extern const tval3_info<double> tval3_cpu_2d(mat<std::complex<double> > &U,
		const mat<std::complex<double> > &A, const mat<std::complex<double> > &b,
		const tval3_options<double> &opts, const mat<std::complex<double> > &Ut, int
		submat_mb, int virtual_procs_per_core) 
	throw(tval3_exception)  __attribute__ ((visibility ("default") ));

extern const tval3_info<float> tval3_cpu_2d(mat<float> &U, const
		sparse_mat<float> &A, const mat<float> &b, const tval3_options<float> &opts,
		const mat<float> &Ut, int submat_mb, int virtual_procs_per_core)
	throw(tval3_exception)  __attribute__ ((visibility ("default") ));

extern const tval3_info<double> tval3_cpu_2d(mat<float> &U, const
		sparse_mat<float> &A, const mat<float> &b, const tval3_options<double>
		&opts, const mat<float> &Ut, int submat_mb, int virtual_procs_per_core)
	throw(tval3_exception)  __attribute__ ((visibility ("default") ));

extern const tval3_info<double> tval3_cpu_2d(mat<double> &U, const
		sparse_mat<double> &A, const mat<double> &b, const tval3_options<double>
		&opts, const mat<double> &Ut, int submat_mb, int virtual_procs_per_core)
	throw(tval3_exception)  __attribute__ ((visibility ("default") ));

extern const tval3_info<float> tval3_cpu_2d(mat<std::complex<float> > &U, const
		sparse_mat<std::complex<float> > &A, const mat<std::complex<float> > &b,
		const tval3_options<float> &opts,  const mat<std::complex<float> > &Ut, 
		int submat_mb, int virtual_procs_per_core) 
	throw(tval3_exception)  __attribute__ ((visibility ("default") ));

extern const tval3_info<double> tval3_cpu_2d(mat<std::complex<double> > &U,
		const sparse_mat<std::complex<double> > &A, const mat<std::complex<double> >
		&b, const tval3_options<double> &opts, const mat<std::complex<double> > &Ut,
		int submat_mb, int virtual_procs_per_core) 
	throw(tval3_exception)  __attribute__ ((visibility ("default") ));

extern const tval3_info<float> tval3_cpu_2d(mat<float> &U, const geometry &A,
		const mat<float> &b, const tval3_options<float> &opts, const mat<float> &Ut,
		int submat_mb, int virtual_procs_per_core) 
	throw(tval3_exception) __attribute__ ((visibility ("default") ));

extern const tval3_info<double> tval3_cpu_2d(mat<float> &U, const geometry &A,
		const mat<float> &b, const tval3_options<double> &opts, const mat<float>
		&Ut, int submat_mb, int virtual_procs_per_core) 
	throw(tval3_exception) __attribute__ ((visibility ("default") ));

extern const tval3_info<double> tval3_cpu_2d(mat<double> &U, const geometry &A,
		const mat<double> &b, const tval3_options<double> &opts, const mat<double>
		&Ut, int submat_mb, int virtual_procs_per_core) 
	throw(tval3_exception) __attribute__ ((visibility ("default") ));

extern const tval3_info<float> tval3_cpu_2d(mat<std::complex<float> > &U, const
		geometry &A, const mat<std::complex<float> > &b, const tval3_options<float>
		&opts, const mat<std::complex<float> > &Ut, int submat_mb, 
		int virtual_procs_per_core)
	throw(tval3_exception)  __attribute__ ((visibility ("default") ));

extern const tval3_info<double> tval3_cpu_2d(mat<std::complex<double> > &U,
		const geometry &A, const mat<std::complex<double> > &b, const
		tval3_options<double> &opts, const mat<std::complex<double> > &Ut, 
		int submat_mb, int virtual_procs_per_core)
	throw(tval3_exception)  __attribute__ ((visibility ("default") ));

// this struct holds various values, accessed by the functions get_g(), update_g(), update_W() and update_mlp()
// the member names correspond to the variable names in the MATLAB-version
template<class T_comp, class T_scal>
struct tval3_data { // parameters and intermediate results...
	// dimensions...
	int p; // #rows of reconstruction "volume"
	int q; // #cols of reconstruction "volume"
	int n; // number of pixels (n = p*q)
	int m; // number of measurements

	// penalty parameters...
	T_scal mu;
	T_scal beta;
	T_scal muDbeta; // mu/beta

	// multiplier...
	mat<T_comp> delta;
	mat<T_comp> sigmaX;
	mat<T_comp> sigmaY;

	// lagrangian function and sub terms...
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
	mat<T_comp> Wx;
	mat<T_comp> Wy;
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

	tval3_data<T_comp, T_scal>(const mat<T_comp> &U, const mat<T_comp> &b, const
			tval3_options<T_scal> &opts): 
		p(U.dim_y), 
		q(U.dim_x), 
		n(U.len), 
		m(b.len),
		mu(opts.mu0), 
		beta(opts.beta0), 
		delta(b.len), 
		sigmaX(U.dim_y, U.dim_x),
		sigmaY(U.dim_y, U.dim_x), 
		Up(U.dim_y, U.dim_x), 
		dU(U.dim_y, U.dim_x),
		uup(U.len), 
		Ux(U.dim_y, U.dim_x), 
		Uxp(U.dim_y, U.dim_x), 
		dUx(U.dim_y, U.dim_x), 
		Uy(U.dim_y, U.dim_x), 
		Uyp(U.dim_y, U.dim_x), 
		dUy(U.dim_y, U.dim_x), 
		Wx(U.dim_y, U.dim_x), 
		Wy(U.dim_y, U.dim_x), 
		Atb(U.len),
		Au(b.len), 
		Aup(b.len), 
		dAu(b.len), 
		d(U.dim_y, U.dim_x), 
		g(U.len),
		gp(U.len), 
		dg(U.len), 
		g2(U.len), 
		g2p(U.len), 
		dg2(U.len),
		numInitIterations(opts.numInitIterations),
		skipMulRatio(opts.skipMulRatio),
		maxRelativeChange(opts.maxRelativeChange), 
		currInitIteration(0),
		currSkipValue(0.0), 
		last_Au(b.len)	{}
};

// z <- x + alpha*y
// N: number of elements
template<class T_mat, class T_scal> 
void vec_add(const int N, const int stride,
		const T_mat *x, const T_scal alpha, const T_mat *y, T_mat *z) {

	#pragma vector always
	#pragma parallel always
	for(int i=0; i<N; i++)
		z[i*stride] = x[i*stride] + alpha*y[i*stride];
}

template<class T_mat, class T_scal> 
void vec_add(const int N, const int stride,
		const std::complex<T_mat> *x, const T_scal alpha, const std::complex<T_mat>
		*y, std::complex<T_mat> *z) {

	T_mat *data_x = (T_mat *)x;
	T_mat *data_y = (T_mat *)y;
	T_mat *data_z = (T_mat *)z;

	#pragma vector always
	#pragma parallel always
	for(int i=0; i<N; i++) {
		data_z[2*i*stride] = data_x[2*i*stride] + alpha*data_y[2*i*stride];
		data_z[2*i*stride + 1] = data_x[2*i*stride + 1] + 
			alpha*data_y[2*i*stride + 1];
	}
}

// Scales the measurement vector. Returns the scale factor.
template<class T_comp, class T_scal>
T_scal scaleb(mat<T_comp> &b) {

	T_scal scl=1;

	if(b.len > 0) {
		T_scal threshold1 = 0.5;
		T_scal threshold2 = 1.5;
		scl=1;
		T_comp val;
		T_scal val_abs;
		T_comp bmin = b[0];
		T_comp bmax = bmin;
		T_scal bmin_abs = abs(bmin);
		T_scal bmax_abs = bmin_abs;
		for(int i=0; i < b.len; i++) {
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
			scl = threshold1/b_dif;
			cblas_scal(b.len, scl, b.data(), 1);
		} else if(b_dif > threshold2) {
			scl = threshold2/b_dif;
			cblas_scal(b.len, scl, b.data(), 1);
		}
	}
	return scl;
}

template<class T_comp, class T_scal>
void D(mat<T_comp> &Ux, mat<T_comp> &Uy, const mat<T_comp> &U) {

	vec_add(U.dim_x*U.dim_y - 1, 1, U.data() + 1, -1, U.data(), Ux.data());
	vec_add(U.dim_y, U.dim_x, U.data(), -1, 
			U.data()+U.dim_x - 1, Ux.data() + U.dim_x - 1);
	vec_add((U.dim_y - 1) * U.dim_x, 1, 
			U.data() + U.dim_x, -1, U.data(), Uy.data());
	vec_add(U.dim_x, 1, U.data(), -1, U.data() + (U.dim_y - 1) * U.dim_x, 
			Uy.data() + (U.dim_y - 1) * U.dim_x);
}

template<class T_comp, class T_scal>
void Dt(mat<T_comp> &res, const mat<T_comp> &X, const mat<T_comp> &Y) {

	vec_add(X.len - 1, 1, X.data(), -1, X.data() + 1, res.data() + 1);
	vec_add(X.dim_y, X.dim_x, X.data()+X.dim_x - 1, -1, X.data(), res.data());

	cblas_axpy((Y.dim_y - 1) * Y.dim_x, (T_comp)1, 
			Y.data(), 1, res.data() + Y.dim_x, 1);
	cblas_axpy((Y.dim_y - 1) * Y.dim_x, (T_comp)-1, 
			Y.data() + Y.dim_x, 1, res.data() + Y.dim_x, 1);
	cblas_axpy(Y.dim_x, (T_comp)1, Y.data() + (Y.dim_y - 1) * Y.dim_x, 
			1, res.data(), 1);
	cblas_axpy(Y.dim_x, (T_comp)-1, Y.data(), 1, res.data(), 1);
}

// overloaded function get_g_mult calculates:
// Au = A * U...
// g = A' * Au...

// standard version (parameter submat_mb not used)
template<class T_comp, class T_scal, class T_A>
void get_g_mult(tval3_data<T_comp, T_scal> &data, const T_A &A, 
		const mat<T_comp> &U, int layers, int submat_mb, bool force_mul = true) {

	gemv(CblasNoTrans, A, U, data.Au, layers);

	if (data.skipMulRatio == 0) {

		gemv(CblasConjTrans, A, data.Au, data.g, layers);

	} else {

		mat<T_comp> deltaAu = data.Au;
		cblas_axpy(deltaAu.len, (T_comp)-1, data.last_Au.data(), 1, deltaAu.data(), 1);
		T_scal nrm_deltaAu = cblas_nrm2(deltaAu.len, deltaAu.data(), 1);
		T_scal nrm_Au = cblas_nrm2(data.Au.len, data.Au.data(), 1);
		T_scal relChange = nrm_deltaAu / nrm_Au;

		if ( force_mul == true ) { 	
			// do always, if forced (used for initialization outside of loop and if
			// deepest descend fails

			data.last_Au = data.Au;
			gemv(CblasConjTrans, A, data.Au, data.g, layers);
			data.currSkipValue = 0;
			//cout << "force\n";

		} else if ( data.currInitIteration < data.numInitIterations ) { 
			// do also for specified number of first iterations, after that do only, if:

			data.last_Au = data.Au;
			gemv(CblasConjTrans, A, data.Au, data.g, layers);
			data.currInitIteration++;
			data.currSkipValue = 0;
			//cout << "init " << data.currInitIteration << "\n";

		} else if ( data.currSkipValue < 1 ) {	 
			// specified skip-ratio is reached or if

			data.last_Au = data.Au;
			gemv(CblasConjTrans, A, data.Au, data.g, layers);
			data.currSkipValue += data.skipMulRatio;
			//cout << "skip " << data.currSkipValue << "\n";

		} else if ( relChange > data.maxRelativeChange) { 
			// data.Au changed to much

			data.last_Au = data.Au;
			gemv(CblasConjTrans, A, data.Au, data.g, layers);
			//cout << "maxChange " << relChange << "\n";

		} else {
			data.currSkipValue -= 1;
			//cout << "..\n";
		}
	}

}

// special version for dense matrices (cache blocking for single layer reconstructions....)
// submat_mb: size of submatrices in MB
template<class T_comp, class T_scal> 
void get_g_mult(tval3_data<T_comp, T_scal> &data, const mat<T_comp> &A, 
		const mat<T_comp> &U, int layers, int submat_mb, bool force_mul = true) {

	if (layers == 1) {
		int submat_elements = submat_mb * 1024 * 1024 / sizeof(T_comp);

		mat<T_comp> g_tmp(data.g.len);

		int submat_rows = max(submat_elements / data.n, 1);

		int n_y = (data.m + submat_rows - 1) / submat_rows;

		int r;
		memset(data.g.data(), 0, data.n * sizeof(T_comp));

		for(int y = 0; y < n_y; y++) {
			r = min(submat_rows, data.m - y * submat_rows);

			cblas_gemv(CblasRowMajor, CblasNoTrans, r, data.n, 1, 
					A.data() + y * submat_rows * data.n, data.n, U.data(), 1, 0, 
					data.Au.data() + y * submat_rows, 1);

			cblas_gemv(CblasRowMajor, CblasConjTrans, r, data.n, 1, 
					A.data() + y * submat_rows * data.n, data.n, 
					data.Au.data() + y * submat_rows, 1, 0, g_tmp.data(), 1);

			cblas_axpy(data.n, 1, g_tmp.data(), 1, data.g.data(), 1);
		}
	} else {
		gemv(CblasNoTrans, A, U, data.Au, layers);
		gemv(CblasConjTrans, A, data.Au, data.g, layers);
	}
}

template<class T_comp, class T_scal, class T_A> 
void get_g(tval3_data<T_comp, T_scal> &data, const T_A &A, const mat<T_comp> &U, 
		const mat<T_comp> &b, int layers, int submat_mb, bool force_mul = true) {

	get_g_mult(data, A, U, layers, submat_mb, force_mul);

	cblas_axpy(data.g.len, (T_comp)-1, data.Atb.data(), 1, data.g.data(), 1);

	// update g2, lam2, lam4...
	mat<T_comp> Vx(data.Ux.dim_y, data.Ux.dim_x);
	mat<T_comp> Vy(data.Uy.dim_y, data.Uy.dim_x);
	vec_add(Vx.len, 1, data.Ux.data(), (T_scal)-1, data.Wx.data(), Vx.data()); // Vx = Ux - Wx
	vec_add(Vx.len, 1, data.Uy.data(), (T_scal)-1, data.Wy.data(), Vy.data()); // Vy = Uy - Wy
	Dt<T_comp, T_scal>(data.g2, Vx, Vy);

	data.lam2=real(cblas_dotc(Vx.len, Vx.data(), 1, Vx.data(), 1));
	data.lam2 += real(cblas_dotc(Vy.len, Vy.data(), 1, Vy.data(), 1));
	data.lam4=real(cblas_dotc(Vx.len, data.sigmaX.data(), 1, Vx.data(), 1));
	data.lam4 += real(cblas_dotc(Vy.len, data.sigmaY.data(), 1, Vy.data(), 1));

	//update lam3, lam5...
	mat<T_comp> Aub(b.len);
	vec_add(b.len, 1, data.Au.data(), (T_scal)-1, b.data(), Aub.data()); // Aub = Au - b
	data.lam3=real(cblas_dotc(Aub.len, Aub.data(), 1, Aub.data(), 1));
	data.lam5=real(cblas_dotc(Aub.len, data.delta.data(), 1, Aub.data(), 1));

	data.f = (data.lam1 + data.beta/2*data.lam2 + data.mu/2*data.lam3) - data.lam4 - data.lam5;
}

template<class T_comp, class T_scal, class T_A>
T_scal get_tau(tval3_data<T_comp, T_scal> &data, const T_A &A, bool fst_iter, int layers) {
	T_scal tau;

	// calculate tau...
	if(fst_iter) {
		mat<T_comp> dx(data.d.dim_y, data.d.dim_x);
		mat<T_comp> dy(data.d.dim_y, data.d.dim_x);
		D<T_comp, T_scal>(dx, dy, data.d);
		T_comp dDd = cblas_dotc(data.d.len, dx.data(), 1, dx.data(), 1) +
					cblas_dotc(data.d.len, dy.data(), 1, dy.data(), 1);
		T_comp dd = cblas_dotc(data.d.len, data.d.data(), 1, data.d.data(), 1);
		mat<T_comp> Ad(data.delta.len);
		// Ad=A*d...
		gemv(CblasNoTrans, A, data.d, Ad, layers);
		T_comp Add = cblas_dotc(Ad.len, Ad.data(), 1, Ad.data(), 1);

		tau = abs(dd/(dDd + data.muDbeta*Add));
	} else {
		vec_add(data.g.len, 1, data.g.data(), -1, data.gp.data(), data.dg.data());
		vec_add(data.g.len, 1, data.g2.data(), -1, data.g2p.data(), data.dg2.data());
		T_comp ss = cblas_dotc(data.uup.len, data.uup.data(), 1, data.uup.data(), 1);
		mat<T_comp> tmp(data.dg2);
		cblas_axpy(tmp.len, data.muDbeta, data.dg.data(), 1, tmp.data(), 1);
		T_comp sy = cblas_dotc(data.uup.len, data.uup.data(), 1, tmp.data(), 1);
		tau = abs(ss/sy);
	}
	return tau;
}

template<class T_mat>
void nonneg(mat<T_mat> &U) {

	T_mat val;
	#pragma parallel always
	#pragma vector always
	for(int i=0; i < U.len; i++) {
		val = U[i];
		U[i] = (val < 0) ? 0 : val;
	}
}

template<class T_mat>
void nonneg(mat<complex<T_mat> > &U) {

	T_mat val;
	T_mat *data = (T_mat *) U.data();
	#pragma parallel always
	#pragma vector always
	for(int i=0; i < U.len; i++) {
		val = data[2*i];
		data[2*i] = (val < 0) ? 0 : val;
		data[2*i + 1] = 0;
	}
}

template<class T_comp, class T_scal, class T_A>
void descend(mat<T_comp> &U, tval3_data<T_comp, T_scal> &data, const T_A &A,
		const mat<T_comp> &b, T_scal tau, bool non_neg, int layers, int submat_mb,
		bool force_mul = true) {

	cblas_axpy(U.len, -1 * tau, data.d.data(), 1, U.data(), 1); // U = U - tau*d
	if(non_neg) {
		nonneg(U);
	}

	D<T_comp, T_scal>(data.Ux, data.Uy, U);
	get_g<T_comp, T_scal>(data, A, U, b, layers, submat_mb, force_mul);
}

template<class T_comp, class T_scal, class T_A>
void update_g(tval3_data<T_comp, T_scal> &data, const T_scal alpha, 
		const T_A &A, mat<T_comp> &U, const mat<T_comp> &b) {

	vec_add(data.g.len, 1, data.gp.data(), alpha, data.dg.data(), data.g.data());
	vec_add(data.g2.len, 1, data.g2p.data(), alpha, data.dg2.data(), data.g2.data());
	vec_add(U.len, 1, data.Up.data(), alpha, data.dU.data(), U.data());
	vec_add(data.Au.len, 1, data.Aup.data(), alpha, data.dAu.data(), data.Au.data());
	vec_add(data.Ux.len, 1, data.Uxp.data(), alpha, data.dUx.data(), data.Ux.data());
	vec_add(data.Uy.len, 1, data.Uyp.data(), alpha, data.dUy.data(), data.Uy.data());

	// update lam2, lam4...
	mat<T_comp> Vx(data.Ux.dim_y, data.Ux.dim_x);
	mat<T_comp> Vy(data.Uy.dim_y, data.Uy.dim_x);
	vec_add(Vx.len, 1, data.Ux.data(), (T_scal)-1, data.Wx.data(), Vx.data()); // Vx = Ux - Wx
	vec_add(Vx.len, 1, data.Uy.data(), (T_scal)-1, data.Wy.data(), Vy.data()); // Vy = Uy - Wy

	data.lam2=real(cblas_dotc(Vx.len, Vx.data(), 1, Vx.data(), 1));
	data.lam2 += real(cblas_dotc(Vy.len, Vy.data(), 1, Vy.data(), 1));
	data.lam4=real(cblas_dotc(Vx.len, data.sigmaX.data(), 1, Vx.data(), 1));
	data.lam4 += real(cblas_dotc(Vy.len, data.sigmaY.data(), 1, Vy.data(), 1));

	//update lam3, lam5...
	mat<T_comp> Aub(b.len);
	vec_add(b.len, 1, data.Au.data(), (T_scal)-1, b.data(), Aub.data()); // Aub = Au - b
	data.lam3 =real(cblas_dotc(Aub.len, Aub.data(), 1, Aub.data(), 1));
	data.lam5=real(cblas_dotc(Aub.len, data.delta.data(), 1, Aub.data(), 1));

	data.f = data.lam1 + data.beta/2*data.lam2 + data.mu/2*data.lam3 - data.lam4 - data.lam5;
}

template<class T_comp, class T_scal, class T_A>
void min_u(tval3_data<T_comp, T_scal> &data, mat<T_comp> &U, T_scal &gam, 
		const T_A &A, const mat<T_comp> &b, const tval3_options<T_scal> &opts, 
		T_comp C, bool fst_iter, int layers, int submat_mb ) {

	T_scal tau, alpha, c_armij;

	tau = get_tau<T_comp, T_scal, T_A>(data, A, fst_iter, layers);

	// keep previous values...
	data.Up = U; data.gp = data.g; data.g2p = data.g2;
	data.Aup = data.Au; data.Uxp = data.Ux; data.Uyp = data.Uy;

	// one step steepest descend...
	descend<T_comp, T_scal, T_A>(U, data, A, b, tau, opts.nonneg, layers, submat_mb, false);

	// NMLA...
	alpha=1;
	vec_add(U.len, 1, U.data(), -1, data.Up.data(), data.dU.data()); // Ud = U - Up
	c_armij = real(cblas_dotc(data.d.len, data.d.data(), 1, data.d.data(), 1) *
			tau * opts.c * data.beta); // c_armij = d'*d*tau*c*beta 

	if(abs(data.f) > abs(C - alpha*c_armij)) {  // Armijo condition

		vec_add(data.g.len, 1, data.g.data(), -1, data.gp.data(), data.dg.data()); // dg=g-gp
		vec_add(data.g.len, 1, data.g2.data(), -1, data.g2p.data(), data.dg2.data()); // dg2=g2-g2p
		vec_add(data.Au.len, 1, data.Au.data(), -1, data.Aup.data(), data.dAu.data()); // dAu=Au-Aup
		vec_add(data.Ux.len, 1, data.Ux.data(), -1, data.Uxp.data(), data.dUx.data()); // dUx=Ux-Uxp
		vec_add(data.Ux.len, 1, data.Uy.data(), -1, data.Uyp.data(), data.dUy.data()); // Uy = Uy-Uyp

		int cnt=0;

		while(abs(data.f) > abs(C - alpha * c_armij)) { // Armijo condition

			// TODO: user smarter value for backtracking check
			if(cnt==5) { // "backtracking" not successful...

				gam *= opts.rate_gam;
				tau = get_tau<T_comp, T_scal, T_A>(data, A, true, layers);
				U = data.Up;
				descend<T_comp, T_scal, T_A>(U, data, A, b, tau, opts.nonneg, layers, submat_mb, true);
				break;
			}
			alpha *= opts.gamma;

			update_g<T_comp, T_scal, T_A>(data, alpha, A, U, b);

			cnt++;
		}
	}
}

template<class T_comp, class T_scal>
void get_gradient(tval3_data<T_comp, T_scal> &data, const mat<T_comp> &DtsAtd) {

	// d = g2 + muDbeta*g + DtsAtd...   (DtsAtd has opposite sign of the MATLAB-version!)
	data.d = DtsAtd;
	cblas_axpy(data.d.len, 1, data.g2.data(), 1, data.d.data(), 1);
	cblas_axpy(data.d.len, data.muDbeta, data.g.data(), 1, data.d.data(), 1);
}

// scalar
template<class T_mat, class T_scal>
void shrinkage(tval3_data<T_mat, T_scal> &data) {

	T_scal sum=0;
	T_scal temp_wx, temp_wy;
	T_mat Uxbar, Uybar;

	#pragma parallel always
	#pragma vector always
	for(int i=0; i < data.Ux.len; i++) {

		Uxbar = data.Ux[i] - data.sigmaX[i]/data.beta;
		Uybar = data.Uy[i] - data.sigmaY[i]/data.beta;

		temp_wx = abs(Uxbar) - 1/data.beta;
		temp_wy = abs(Uybar) - 1/data.beta;
		temp_wx = (temp_wx >= 0) ? temp_wx : 0;
		temp_wy = (temp_wy >= 0) ? temp_wy : 0;
		sum += temp_wx + temp_wy;
		data.Wx[i] = (Uxbar >= 0) ? temp_wx : -1 * temp_wx;
		data.Wy[i] = (Uybar >= 0) ? temp_wy : -1 * temp_wy;
	}
	data.lam1 = sum;
}

// complex
template<class T_mat, class T_scal>
void shrinkage(tval3_data<complex<T_mat>, T_scal> &data) {

	T_mat sum=0;
	T_mat temp_wx, temp_wy;
	T_mat Uxbar_real, Uxbar_imag, Uxbar_abs, Uybar_real, Uybar_imag, Uybar_abs;
	T_mat *Ux_data = (T_mat *) data.Ux.data();
	T_mat *Uy_data = (T_mat *) data.Uy.data();
	T_mat *sigmaX_data = (T_mat *) data.sigmaX.data();
	T_mat *sigmaY_data = (T_mat *) data.sigmaY.data();
	T_mat *Wx_data = (T_mat *) data.Wx.data();
	T_mat *Wy_data = (T_mat *) data.Wy.data();
	const int len = data.Ux.len;

	#pragma parallel always
	#pragma vector always
	for(int i=0; i < len; i++) {
		Uxbar_real = Ux_data[2*i] - sigmaX_data[2*i]/data.beta;
		Uxbar_imag = Ux_data[2*i + 1] - sigmaX_data[2*i + 1]/data.beta;
		Uybar_real = Uy_data[2*i] - sigmaY_data[2*i]/data.beta;
		Uybar_imag = Uy_data[2*i + 1] - sigmaY_data[2*i + 1]/data.beta;

		Uxbar_abs = sqrt(Uxbar_real*Uxbar_real + Uxbar_imag*Uxbar_imag);
		Uybar_abs = sqrt(Uybar_real*Uybar_real + Uybar_imag*Uybar_imag);
		temp_wx = Uxbar_abs - 1/data.beta;
		temp_wy = Uybar_abs - 1/data.beta;
		temp_wx = (temp_wx >= 0) ? temp_wx : 0;
		temp_wy = (temp_wy >= 0) ? temp_wy : 0;
		sum += temp_wx + temp_wy;
		Wx_data[2*i] = (Uxbar_abs > 0) ? temp_wx*Uxbar_real/Uxbar_abs : 0;
		Wx_data[2*i+1] = (Uxbar_abs > 0) ? temp_wx*Uxbar_imag/Uxbar_abs : 0;
		Wy_data[2*i] = (Uybar_abs > 0) ? temp_wy*Uybar_real/Uybar_abs : 0;
		Wy_data[2*i+1] = (Uybar_abs > 0) ? temp_wy*Uybar_imag/Uybar_abs : 0;
	}
	data.lam1 = sum;
}

template<class T_comp, class T_scal>
void update_W(tval3_data<T_comp, T_scal> &data) {

	data.f -= (data.lam1 + data.beta/2*data.lam2 - data.lam4);

	shrinkage(data);

	// update g2, lam2, lam4...
	mat<T_comp> Vx(data.Ux.dim_y, data.Ux.dim_x);
	mat<T_comp> Vy(data.Uy.dim_y, data.Uy.dim_x);
	vec_add(Vx.len, 1, data.Ux.data(), (T_scal)-1, data.Wx.data(), Vx.data()); // Vx = Ux - Wx
	vec_add(Vx.len, 1, data.Uy.data(), (T_scal)-1, data.Wy.data(), Vy.data()); // Vy = Uy - Wy
	Dt<T_comp, T_scal>(data.g2, Vx, Vy);

	data.lam2=real(cblas_dotc(Vx.len, Vx.data(), 1, Vx.data(), 1));
	data.lam2 += real(cblas_dotc(Vy.len, Vy.data(), 1, Vy.data(), 1));
	data.lam4=real(cblas_dotc(Vx.len, data.sigmaX.data(), 1, Vx.data(), 1));
	data.lam4 += real(cblas_dotc(Vy.len, data.sigmaY.data(), 1, Vy.data(), 1));

	data.f += (data.lam1 + data.beta/2*data.lam2 - data.lam4);
}

template<class T_comp, class T_scal>
void update_mlp(tval3_data<T_comp, T_scal> &data, const mat<T_comp> &b) {
	data.f += (data.lam4 + data.lam5);

	mat<T_comp> Vx(data.Ux.dim_y, data.Ux.dim_x);
	mat<T_comp> Vy(data.Uy.dim_y, data.Uy.dim_x);
	vec_add(Vx.len, 1, data.Ux.data(), (T_scal)-1, data.Wx.data(), Vx.data()); // Vx = Ux - Wx
	vec_add(Vx.len, 1, data.Uy.data(), (T_scal)-1, data.Wy.data(), Vy.data()); // Vy = Uy - Wy

	cblas_axpy(data.sigmaX.len, -1*data.beta, Vx.data(), 1, data.sigmaX.data(), 1); // sigmaX -= beta*Vx
	cblas_axpy(data.sigmaY.len, -1*data.beta, Vy.data(), 1, data.sigmaY.data(), 1); // sigmaY -= beta*Vy

	data.lam4=real(cblas_dotc(Vx.len, data.sigmaX.data(), 1, Vx.data(), 1));
	data.lam4 += real(cblas_dotc(Vy.len, data.sigmaY.data(), 1, Vy.data(), 1));

	mat<T_comp> Aub(b.len);
	vec_add(b.len, 1, data.Au.data(), (T_scal)-1, b.data(), Aub.data()); // Aub = Au - b
	cblas_axpy(data.delta.len, -1*data.mu, Aub.data(), 1, data.delta.data(), 1); // delta -= mu*Aub
	data.lam5=real(cblas_dotc(Aub.len, data.delta.data(), 1, Aub.data(), 1));

	data.f -= (data.lam4 + data.lam5);
}

template<class T_comp>
void check_params(mat<T_comp> &U, const mat<T_comp> &A, const mat<T_comp> &b,
		const mat<T_comp> &Ut, int layers) throw(tval3_exception) { 
	
	if(U.format !=
			mat_col_major || Ut.format != mat_col_major || A.format != mat_row_major)
		throw tval3_exception(string("Argument error: Matrices must be in row major format!"));

	else if(Ut.len > 0 && Ut.len != U.len)
		throw tval3_exception(string("Argument error: U and Ut must have the same size!"));

	else if(U.len != A.dim_x * layers)
		throw tval3_exception("Argument error: the length of U must be equal to A.dim_x * layers!");

	else if(b.len != A.dim_y * layers)
		throw tval3_exception(string("Argument error: b.len must be equal to A.dim_y * layers!"));

}

template<class T_comp>
void check_params(mat<T_comp> &U, const sparse_mat<T_comp> &A, const mat<T_comp>
		&b, const mat<T_comp> &Ut, int layers) throw(tval3_exception) {

	if(U.format != mat_col_major || Ut.format != mat_col_major)
		throw tval3_exception(string("Argument error: Matrices must be in row major format!"));

	if(A.format != sparse_mat_both)
		throw tval3_exception(string("Argument error: A must be available in CSR and CSC format!"));

	if(Ut.len > 0 && Ut.len != U.len)
		throw tval3_exception(string("Argument error: U and Ut must have the same size!"));

	else if(U.len != A.dim_x * layers)
		throw tval3_exception("Argument error: the length of U must be equal to A.dim_x * layers!");

	else if(b.len != A.dim_y * layers)
		throw tval3_exception(string("Argument error: b.len must be equal to A.dim_y * layers!"));
}

template<class T_comp>
void check_params(mat<T_comp> &U, const geometry &A, const mat<T_comp> &b, const
		mat<T_comp> &Ut, int layers) throw(tval3_exception) {

	if(U.format != mat_row_major || Ut.format != mat_row_major)
		throw tval3_exception(string("Argument error: Matrices must be in row major format!"));

	else if(Ut.len > 0 && Ut.len != U.len)
		throw tval3_exception(string("Argument error: U and Ut must have the same size!"));

	else if(layers > 1)
		throw tval3_exception(string("Argument error: multilayer reconstruction with dynamic calculation is not supported!"));

	else if(b.len != A.num_emitters * A.num_receivers)
		throw tval3_exception(string("Argument error: b.len must be equal to A.num_emitters * A.num_receivers!"));
}

template<class T_comp, class T_scal, class T_A>
const tval3_info<T_scal> tval3_cpu_2d(mat<T_comp> &U, const T_A &A, const
		mat<T_comp> &b, const tval3_options<T_scal> &opts, const mat<T_comp> &Ut, 
		int layers, int submat_mb=5, int virtual_procs_per_core=2) 
	throw(tval3_exception) {

	double start, stop;
	tval3_info<T_scal> info;
	start = get_timestamp();
	try {

		check_params (U, A, b, Ut, layers);

		mkl_set_num_threads(omp_get_num_procs() / virtual_procs_per_core);

		mat<T_comp> bs = b; // create a copy of b (scaling...)
		tval3_data<T_comp, T_scal> data(U, bs, opts);
		T_scal muf=opts.mu;
		T_scal betaf=opts.beta, beta0=0;
		mat<T_comp> DtsAtd(data.d.len);
		T_scal gam = opts.gam;
		T_scal scl = scaleb<T_comp, T_scal>(bs);
		gemv(CblasConjTrans, A, bs, data.Atb, layers); // Atb = A'*bs
		U = data.Atb; // initial guess: U=A'*b
		if(data.mu > muf) data.mu = muf;
		if(data.beta > betaf) data.beta = betaf;
		data.muDbeta = data.mu/data.beta;
		mat <T_comp> rcdU(U), UrcdU(U.len);
		D<T_comp, T_scal>(data.Ux, data.Uy, U);
		shrinkage(data);
		get_g<T_comp, T_scal>(data, A, U, bs, layers, submat_mb, true);
		get_gradient<T_comp, T_scal>(data, DtsAtd);
		T_scal Q=1, Qp;
		T_comp C=data.f;
		int count_outer=0, count_total=0;
		bool fst_iter;
		T_scal RelChg, RelChgOut=0, nrmup=0;

		while((count_outer < opts.maxcnt) && (count_total < opts.maxit)) {

			fst_iter = true;
			while(count_total < opts.maxit) {
				// ui subproblem
				min_u<T_comp, T_scal>(data, U, gam, A, bs, opts, C, fst_iter, 
						layers, submat_mb);

				// shrinkage like step
				update_W<T_comp, T_scal>(data);

				// update reference values
				Qp = Q, Q = gam*Qp + 1; C = (gam*Qp*C + data.f)/Q;
				vec_add(U.len, 1, U.data(), -1, data.Up.data(), data.uup.data());

				// compute gradient
				get_gradient<T_comp, T_scal>(data, DtsAtd);

				count_total++;

				// calculate relative change
				nrmup = cblas_nrm2(data.Up.len, data.Up.data(), 1);
				RelChg = cblas_nrm2(data.uup.len, data.uup.data(), 1) / nrmup;

				if((RelChg < opts.tol_inn) && (!fst_iter))
					break;

				fst_iter = false;

			}
			count_outer++;

			// calculate relative change
			vec_add(U.len, 1, U.data(), -1, rcdU.data(), UrcdU.data()); // UrcdU = U - rcdU
			RelChgOut = cblas_nrm2(UrcdU.len, UrcdU.data(), 1) / nrmup;
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
			data.muDbeta = data.mu/data.beta;

			// update f
			data.f = data.lam1 + data.beta/2*data.lam2 + data.mu/2*data.lam3 - data.lam4 - data.lam5;

			// DtsAtd = beta0/beta*d
			DtsAtd = data.d;
			cblas_scal(DtsAtd.len, beta0/data.beta, DtsAtd.data(), 1);

			// compute gradient
			get_gradient<T_comp, T_scal>(data, DtsAtd);

			// reset reference values
			gam=opts.gam; Q=1; C=data.f;
		}

		// scale U. Take real part only, if opts.isreal == true
		for(int i=0; i < U.len; i++) {
			if(opts.isreal)
				U[i] = (T_comp)real(U[i]);
			U[i] = U[i]/scl;
		}

		stop = get_timestamp();

		// generate return value
		info.secs = stop - start;
		info.outer_iters = count_outer;
		info.total_iters = count_total;
		info.rel_chg = RelChgOut;
		if(Ut.len == U.len) {
			mat<T_comp> deltaU = Ut;
			cblas_axpy(Ut.len, (T_comp)-1, U.data(), 1, deltaU.data(), 1);
			T_scal nrm_deltaU = cblas_nrm2(deltaU.len, deltaU.data(), 1);
			T_scal nrm_Ut = cblas_nrm2(Ut.len, Ut.data(), 1);
			info.rel_error = nrm_deltaU/nrm_Ut;
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
	return  info;
}


const tval3_info<float> tval3_cpu_2d(mat<float> &U, const mat<float> &A, const
		mat<float> &b, const tval3_options<float> &opts, const mat<float> &Ut, int
		submat_mb=5, int virtual_procs_per_core=2) throw(tval3_exception) {

	return tval3_cpu_2d<float, float, mat<float> >(U, A, b, opts, 
			Ut, b.len / A.dim_y, submat_mb, virtual_procs_per_core);
}

const tval3_info<double> tval3_cpu_2d(mat<float> &U, const mat<float> &A, const
		mat<float> &b, const tval3_options<double> &opts, const mat<float> &Ut, int
		submat_mb=5, int virtual_procs_per_core=2) throw(tval3_exception) {

	return tval3_cpu_2d<float, double, mat<float> >(U, A, b, opts, Ut, 
			b.len / A.dim_y, submat_mb, virtual_procs_per_core);
}

const tval3_info<double> tval3_cpu_2d(mat<double> &U, const mat<double> &A,
		const mat<double> &b, const tval3_options<double> &opts, const mat<double>
		&Ut, int submat_mb=5, int virtual_procs_per_core=2) throw(tval3_exception) {

	return tval3_cpu_2d<double, double, mat<double> >(U, A, b, opts, 
			Ut, b.len / A.dim_y, submat_mb, virtual_procs_per_core);
}

const tval3_info<float> tval3_cpu_2d(mat<complex<float> > &U, const
		mat<complex<float> > &A, const mat<complex<float> > &b, const
		tval3_options<float> &opts, const mat<complex<float> > &Ut, int submat_mb=5,
		int virtual_procs_per_core=2) throw(tval3_exception) {

	return tval3_cpu_2d<std::complex<float>, float, mat<std::complex<float> > >(
			U, A, b, opts, Ut, b.len / A.dim_y, submat_mb, virtual_procs_per_core);
}

const tval3_info<double> tval3_cpu_2d(mat<complex<double> > &U, const
		mat<complex<double> > &A, const mat<complex<double> > &b, const
		tval3_options<double> &opts, const mat<complex<double> > &Ut, int
		submat_mb=5, int virtual_procs_per_core=2) throw(tval3_exception) {

	return tval3_cpu_2d<std::complex<double>, double, mat<std::complex<double> > >(
			U, A, b, opts, Ut, b.len / A.dim_y, submat_mb, virtual_procs_per_core);
}


const tval3_info<float> tval3_cpu_2d(mat<float> &U, const sparse_mat<float> &A,
		const mat<float> &b, const tval3_options<float> &opts, const mat<float> &Ut,
		int submat_mb=5, int virtual_procs_per_core=2) throw(tval3_exception) {

	return tval3_cpu_2d<float, float, sparse_mat<float> >(U, A, b, opts, 
			Ut, b.len / A.dim_y, submat_mb, virtual_procs_per_core);
}

const tval3_info<double> tval3_cpu_2d(mat<float> &U, const sparse_mat<float> &A,
		const mat<float> &b, const tval3_options<double> &opts, const mat<float>
		&Ut, int submat_mb=5, int virtual_procs_per_core=2) throw(tval3_exception) {

	return tval3_cpu_2d<float, double, sparse_mat<float> >(U, A, b, opts, 
			Ut, b.len / A.dim_y, submat_mb, virtual_procs_per_core);
}

const tval3_info<double> tval3_cpu_2d(mat<double> &U, 
		const sparse_mat<double> &A, const mat<double> &b, 
		const tval3_options<double> &opts, const mat<double> &Ut, 
		int submat_mb=5, int virtual_procs_per_core=2) throw(tval3_exception) {

	return tval3_cpu_2d<double, double, sparse_mat<double> >(U, A, b, opts, 
			Ut, b.len / A.dim_y, submat_mb, virtual_procs_per_core);
}

const tval3_info<float> tval3_cpu_2d(mat<complex<float> > &U, 
		const sparse_mat<complex<float> > &A, const mat<complex<float> > &b, 
		const tval3_options<float> &opts, const mat<complex<float> > &Ut, 
		int submat_mb=5, int virtual_procs_per_core=2) throw(tval3_exception) {

	return tval3_cpu_2d<std::complex<float>, float, sparse_mat<std::complex<float> > >(
			U, A, b, opts, Ut, b.len / A.dim_y, submat_mb, virtual_procs_per_core);
}

const tval3_info<double> tval3_cpu_2d(mat<complex<double> > &U, const
		sparse_mat<complex<double> > &A, const mat<complex<double> > &b, const
		tval3_options<double> &opts, const mat<complex<double> > &Ut, int
		submat_mb=5, int virtual_procs_per_core=2) throw(tval3_exception) {

	return tval3_cpu_2d<std::complex<double>, double, sparse_mat<std::complex<double> > >(
			U, A, b, opts, Ut, b.len / A.dim_y, submat_mb, virtual_procs_per_core); 
}

const tval3_info<float> tval3_cpu_2d(mat<float> &U, const geometry &A, 
		const mat<float> &b, const tval3_options<float> &opts, const mat<float> &Ut, 
		int submat_mb=5, int virtual_procs_per_core=2) throw(tval3_exception) {

	return tval3_cpu_2d<float, float, geometry>(U, A, b, opts, 
			Ut, 1, submat_mb, virtual_procs_per_core);
}

const tval3_info<double> tval3_cpu_2d(mat<float> &U, const geometry &A, 
		const mat<float> &b, const tval3_options<double> &opts, const mat<float> &Ut, 
		int submat_mb=5, int virtual_procs_per_core=2) throw(tval3_exception) {

	return tval3_cpu_2d<float, double, geometry>(U, A, b, opts, 
			Ut, 1, submat_mb, virtual_procs_per_core);
}

const tval3_info<double> tval3_cpu_2d(mat<double> &U, const geometry &A, 
		const mat<double> &b, const tval3_options<double> &opts, 
		const mat<double> &Ut, int submat_mb=5, 
		int virtual_procs_per_core=2) throw(tval3_exception) {

	return tval3_cpu_2d<double, double, geometry>(U, A, b, opts, 
			Ut, 1, submat_mb, virtual_procs_per_core);
}

const tval3_info<float> tval3_cpu_2d(mat<complex<float> > &U, const geometry &A,
		const mat<complex<float> > &b, const tval3_options<float> &opts, 
		const mat<complex<float> > &Ut, int submat_mb=5,
		int virtual_procs_per_core=2) throw(tval3_exception) {

	return tval3_cpu_2d<std::complex<float>, float, geometry>(U, A, b, opts, 
			Ut, 1, submat_mb, virtual_procs_per_core);
}

const tval3_info<double> tval3_cpu_2d(mat<complex<double> > &U, 
		const geometry &A, const mat<complex<double> > &b, 
		const tval3_options<double> &opts, const mat<complex<double> > &Ut, 
		int submat_mb=5, int virtual_procs_per_core=2) throw(tval3_exception) {

	return tval3_cpu_2d<std::complex<double>, double, geometry>(U, A, b, opts, 
			Ut, 1, submat_mb, virtual_procs_per_core);
}
