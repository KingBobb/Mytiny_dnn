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

	};


	/**
	* class containing input/output data
	**/


}
#endif // !NODE_H
