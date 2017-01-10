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

#include"../config.h"

#include"aligned_allocator.h"


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


    
	//未完待续
	template <typename T> inline T sqr(T value){ return value*value; }

	inline bool isfinite(float_t x) {
		return x == x;
	}

	template <typename Container> 
	inline bool has_infinite(const Container& c) {
		for (auto v : c)
			if (!isfinite(v)) return true;
		return false;
	}

	template <typename Container>
	size_t max_size(const Container& c) {
		typedef typename Container::value_type value_t;
		return std::max_element(c.begin(), c.end(),
			[](const value_t& left, const value_t& right) { return left.size() < right.size(); })->size();
	}

	inline std::string format_str(const char *fmt, ...) {
		static char buf[2048];

#ifdef _MSC_VER
#pragma warning(disable:4996)
#endif
		va_list args;
		va_start(args, fmt);
		vsnprintf(buf, sizeof(buf), fmt, args);
		va_end(args);
#ifdef _MSC_VER
#pragma warning(default:4996)
#endif
		return std::string(buf);
	}

	/********一个三元结构体模板********/
    template <typename T>
	struct index3d{
		index3d(T width, T height, T depth)
		{
			reshape(width, height, depth);
		}

		void reshape(T width, T height, T depth)
		{
			width_ = width;
			height_ = height;
			depth_ = depth;

			if ((long long)width * height * depth > std::numeric_limits<T>::max())
				throw nn_error(
				format_str("error while constructing layer: layer size too large for tiny-dnn\nWidthxHeightxChannels=%dx%dx%d >= max size of [%s](=%d)",
				width, height, depth, typeid(T).name(), std::numeric_limits<T>::max()));
		}
	
		T get_index(T x, T y, T channel)const{
			assert(x >= 0 && x < width_);
			assert(y >= 0 && y < height_);
			assert(channel >= 0 && channel < depth_);
			return (height_ * channel + y) * width_ + x;
		}

		T area() const {
			return width_ * height_;
		}

		T size() const {
			return width_ * height_ * depth_;
		}

		template <class Archive>
		void serialize(Archive & ar) {
			ar(cereal::make_nvp("width", width_));
			ar(cereal::make_nvp("height", height_));
			ar(cereal::make_nvp("depth", depth_));
		}

		T width_;
		T height_;
		T depth_;
	};

	




}


#endif // !UTIL_H
