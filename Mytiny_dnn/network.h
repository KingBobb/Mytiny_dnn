#ifndef NETWORK_H
#define NETWORK_H

#include<iostream>
#include<stdexcept>
#include<algorithm>
#include<iomanip>
#include<iterator>
#include<map>
#include<set>
#include<vector>
#include<string>


#include"nodes.h"
#include"util/util.h"
#include"loss_function.h"
#include"activation_function.h"


namespace mytiny_dnn
{
	enum class content_type{
		weights,   ///< save/load the weights
		model,     ///< save/load the network architecture
		weights_and_model     ///< save/load both the weights and the architecture
	};

	enum class file_format {
		binary,
		json
	};

	struct result{
		int num_success;
		int num_total;
		//map<

		result() :num_success(0), num_total(0){}

		float_t accuracy()const{
			return float_t(num_success*100.0 / num_total);
		}

		template <typename Char,typename CharTraits>
		void print_summary(std::basic_ostream<Char, CharTraits>& os)const{
			os << "accuracy:" << accuracy()
				<< "% (" << num_success << "/"
				<< num_total << ")" << std::endl;
		}

		template <typename Char, typename CharTraits>
		void print_detail(std::basic_ofstream<Char, CharTraits>& os){
			print_summary(os);
			auto all_labels = labels();

			os << std::setw(5) << "*" << " ";
			for (auto c : all_labels)
				os << std::setw(5) << c << " ";
			os << std::endl;

			for (auto r : all_labels) {
				os << std::setw(5) << r << " ";
				for (auto c : all_labels)
					os << std::setw(5) << confusion_matrix[r][c] << " ";
				os << std::endl;
			}
		}

		std::set<label_t> labels()const{
			std::set<label_t>all_labels;
			for (auto r : confusion_matrix)
			{
				all_labels.insert(r.first);
				for (auto c : r.second)
					all_labels.insert(c.first);
			}
			return all_labels;
		}
		std::map<label_t, std::map<label_t, int>>confusion_matrix;
		
	};

	enum grad_check_mode{
		GRAD_CHECK_ALL,    ///< check all elements of weights
		GRAD_CHEVK_RANDOM  ///< check 10 randomly selected weights
	};
	
	template<typename NetType>
	class network;

	//template<typename Layer>
	//Î´Íê´ýÐø
	

}
#endif // !NETWORK_H

