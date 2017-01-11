#ifndef NODE_H
#define NODE_H
#include <sstream>
#include <iomanip>
#include <memory>
#include <numeric>
#include <vector>
#include <set>
#include <queue>
#include <unordered_set>

#include"util\util.h"
#include"util\image.h"
#include"util\weight_init.h"
#include"util\product.h"

#include"optimizers\optimizer.h"

#include"activations\activation_function.h"

namespace mytiny_dnn{

	class node;
	class layer;
	class edge;
	

	typedef node* nodeptr_t;
	typedef std::shared_ptr<edge> edgeptr_t;

	typedef layer* layerptr_t;

	/**
	* base class of all kind of tinny-cnn data
	**/
//	class node:public std::enabled_shared_from
	class node :public std::enable_shared_from_this<node>{
	public:
		node(cnn_size_t in_size,cnn_size_t out_size)
			:prev_(in_size), next_(out_size){}
		virtual ~node(){}
	protected:
		node() = delete;

		friend void connect(layerptr_t head, layerptr_t tail,
			cnn_size_t head_index, cnn_size_t tail_index);

		cnn_size_t prev_port(const edge& e)const{
			auto it = std::find_if(prev_.begin(), prev_.end(), [&](edgeptr_t ep) { return ep.get() == &e; });
			return (cnn_size_t)std::distance(prev_.begin(), it);
		}

		cnn_size_t next_port(const edge& e)const{
			auto it = std::find_if(prev_.begin(), prev_.end(), [&](edgeptr_t ep) { return ep.get() == &e; });

		}
		mutable std::vector<edgeptr_t> prev_;
		mutable std::vector<edgeptr_t> next_;	
	};


    



	/**
	* class containing input/output data
	**/


}
#endif // !NODE_H
