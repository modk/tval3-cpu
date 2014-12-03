#ifndef UTILITY_H_
#define UTILITY_H_

#include <sys/types.h>
#include <cstdlib>
#include <sys/time.h>

/**********************************************************************************/
/* functions, etc. for handling different data types in mathematical expressions  */
/**********************************************************************************/

// overloaded versions of conj() and real() for scalar values...
// (not needed when compiled with gcc and -std_c++0x)

template <class Type>
inline Type conj(Type f) {
	return f;
}

template<class Type>
inline Type real(Type f) {
	return f;
}

template<class Type>
inline Type imag(Type f) {
	return 0;
}


// sign function
template<class T_comp, class T_scal>
inline T_comp sign(T_comp x) {

	T_scal ax = std::abs(x);
	if(ax > 0)
		return x/ax;
	else
		return 0;
}

// get time stamp in seconds
inline double get_timestamp() {

	struct timeval tp;
	double sec, usec;
	gettimeofday(&tp, NULL);
	sec = static_cast<double>(tp.tv_sec);
	usec = static_cast<double>(tp.tv_usec) / 1e6;
	return sec + usec;
}

#endif /* UTILITY_H_ */
