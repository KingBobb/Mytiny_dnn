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

	typedef index3d<cnn_size_t> shape3d;

	template<typename T>
	bool operator ==(const index3d<T>& lhs, const index3d<T>& rhs){
		return (lhs.width_ == rhs.width_) && (lhs.height_ == rhs.height_) && (lhs.depth_ == rhs.depth_);
	}

	template <typename T>
	bool operator != (const index3d<T>& lhs, const index3d<T>& rhs) {
		return !(lhs == rhs);
	}

	template<typename Stream,typename T>
	Stream& operator<<(Stream& s, const std::vector<index3d<T>>& d){
		s << "[";
		for (cnn_size_t i = 0; i < d.size(); i++)
		{
			if (i) s << ",";
			s << "[" << d[i] << "]";
		}
		s << "]";
		return s;
	}


	// equivalent to std::to_string, which android NDK doesn't support
	template <typename T>
	std::string to_string(T value) {
		std::ostringstream os;
		os << value;
		return os.str();
	}


    //未完待续

	template <typename T, typename Pred, typename Sum>
	cnn_size_t sumif(const std::vector<T>& vec, Pred p, Sum s) {
		size_t sum = 0;
		for (size_t i = 0; i < vec.size(); i++) {
			if (p(i)) sum += s(vec[i]);
		}
		return sum;
	}

	template <typename T,typename Pred>
	std::vector<T> filter(const std::vector<T>& vec, Pred p){
		std::vector<T> res;
		for (size_t i = 0; i < vec.size(); i++){
			if (p(i)) res.push_back(vec[i]);
		}
		return res;
	}

	template<typename Result,typename T,typename Pred>
	std::vector<Result> map_(const std::vector<Result>& vec, Pred p){
		std::vector<Result> res;
		for (auto& v : vec){
			res.push_back(p(v));
		}
		return res;
	}


	enum class vector_type : int32_t {
		// 0x0001XXX : in/out data
		data = 0x0001000, // input/output data, fed by other layer or input channel

		// 0x0002XXX : trainable parameters, updated for each back propagation
		weight = 0x0002000,
		bias = 0x0002001,

		label = 0x0004000,
		aux = 0x0010000 // layer-specific storage
	};

	inline std::string to_string(vector_type vtype){
		switch (vtype)
		{
		case mytiny_dnn::vector_type::data:
			return "data";
		case mytiny_dnn::vector_type::weight:
			return "weight";
		case mytiny_dnn::vector_type::bias:
			return "bias";
		case mytiny_dnn::vector_type::label:
			return "label";
		case mytiny_dnn::vector_type::aux:
			return "aux";
		default:
			return "unknown";
		}
	}

	inline std::ostream& operator << (std::ostream& os, vector_type vtype){
		os << to_string(vtype);
		return os;
	}

	inline vector_type operator & (vector_type lhs, vector_type rhs) {
		return (vector_type)(static_cast<int32_t>(lhs)& static_cast<int32_t>(rhs));
	}

	inline bool is_trainable_weight(vector_type vtype){
		return (vtype&vector_type::weight) == vector_type::weight;
	}

	inline std::vector<vector_type> std_input_order(bool has_bias) {
		if (has_bias) {
			return{ vector_type::data, vector_type::weight, vector_type::bias };
		}
		else {
			return{ vector_type::data, vector_type::weight };
		}
	}

	inline std::vector<vector_type> std_output_order(bool has_activation) {
		if (has_activation) {
			return{ vector_type::data, vector_type::aux };
		}
		else {
			return{ vector_type::data };
		}
	}


}


#endif // !UTIL_H
