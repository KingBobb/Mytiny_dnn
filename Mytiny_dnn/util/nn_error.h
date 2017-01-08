#ifndef NN_ERROR_H
#define NN_ERROR_H

#include<exception>
#include<string>


namespace mytiny_dnn{
	class nn_error :public std::exception{
	public:
		explicit nn_error(const std::string& msg) :msg_(msg){};
		const char* what()const throw()override{
			return msg_.c_str();
		}

	private:
		std::string msg_;
	};
} // namespace tiny_dnn

#endif