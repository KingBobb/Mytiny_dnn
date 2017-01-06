#ifndef ALIGNED_ALLOCATOR_H
#define ALIGNED_ALLOCATOR_H

#include<stdlib.h>
#ifdef _WIN32
#include <malloc.h>
#endif
#ifdef __MINGW32__
#include <mm_malloc.h>
#endif
#include "nn_error.h"


namespace mytiny_dnn{

	template <typename T,std::size_t alignment>
	class aligned_allocator{
	public:
		typedef T value_type;
		typedef T* pointer;
		typedef std::size_t size_type;
		typedef std::ptrdiff_t difference_type;
		typedef T& reference;
		typedef const T& const_reference;
		typedef const T* const_pointer;

		template <typename U>
		struct rebind{
			typedef aligned_allocator<U, alignment> other;
		};


	};

}
#endif // !ALIGNED_ALLOCATOR_H
