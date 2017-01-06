#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <functional>
#include <random>
#include <type_traits>
#include <limits>
#include <cassert>
#include <cstdio>
#include <cstdarg>
#include <string>
#include <sstream>

#include"config.h"

#include"util/aligned_allocator.h"


namespace mytiny_dnn{
	///< output label(class-index) for classification
	///< must be equal to cnn_size_t, because size of last layer is equal to num. of classes
	typedef cnn_size_t label_t;

	typedef cnn_size_t layer_size_t; // for backward compatibility

	typedef std::vector<float_t, aligned_allocator<float_t, 64>> vec_t;

	typedef std::vector<vec_t> tensor_t;

	enum class net_phase {
		train,
		test
	};

	template<typename T>
	T* reverse_endian(T* p) {
		std::reverse(reinterpret_cast<char*>(p), reinterpret_cast<char*>(p)+sizeof(T));
		return p;
	}

	inline bool is_little_endian() {
		int x = 1;
		return *(char*)&x != 0;
	}

	template<typename T>
	size_t max_index(const T& vec) {
		auto begin_iterator = std::begin(vec);
		return std::max_element(begin_iterator, std::end(vec)) - begin_iterator;
	}

	template<typename T, typename U>
	U rescale(T x, T src_min, T src_max, U dst_min, U dst_max) {
		U value = static_cast<U>(((x - src_min) * (dst_max - dst_min)) / (src_max - src_min) + dst_min);
		return std::min(dst_max, std::max(value, dst_min));
	}

	inline void nop()
	{
		// do nothing
	}

	//Î´Íê´ýÐø

}


#endif // !UTIL_H
