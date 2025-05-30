#ifndef HGCALML_MODULES_COMPILED_CUDA_DEVICE_HELPERS_H_
#define HGCALML_MODULES_COMPILED_CUDA_DEVICE_HELPERS_H_

#ifdef __CUDACC__

template <class T>
class _grid_stride_range {
public:
    __device__ _grid_stride_range(T ibegin_, T iend_) : ibegin(ibegin_), iend(iend_) {}

    class iterator {
    public:
        __device__ iterator() {}
        __device__ iterator(T pos_) : pos(pos_) {}

        __device__ T operator*() const { return pos; }

        __device__ iterator& operator++() {
            pos += gridDim.x * blockDim.x;
            return *this;
        }

        __device__ bool operator!=(const iterator& item) const { return pos < item.pos; }

    private:
        T pos;
    };

    __device__ iterator begin() const {
        return iterator(ibegin + blockDim.x * blockIdx.x + threadIdx.x);
    }
    __device__ iterator end() const {
        return iterator(iend);
    }

private:
    T ibegin;
    T iend;
};

template <class T>
class _grid_stride_range_y {
public:
    __device__ _grid_stride_range_y(T ibegin_, T iend_) : ibegin(ibegin_), iend(iend_) {}

    class iterator {
    public:
        __device__ iterator() {}
        __device__ iterator(T pos_) : pos(pos_) {}

        __device__ T operator*() const { return pos; }

        __device__ iterator& operator++() {
            pos += gridDim.y * blockDim.y;
            return *this;
        }

        __device__ bool operator!=(const iterator& item) const { return pos < item.pos; }

    private:
        T pos;
    };

    __device__ iterator begin() const {
        return iterator(ibegin + blockDim.y * blockIdx.y + threadIdx.y);
    }
    __device__ iterator end() const {
        return iterator(iend);
    }

private:
    T ibegin;
    T iend;
};

template<class T>
class _lock {
public:
    __device__ _lock(T* mutex) : mutex_(mutex) {}
    __device__ void lock() {
        while(atomicCAS(mutex_, 0, 1) != 0) {}
    }
    __device__ void unlock() {
        atomicExch(mutex_, 0);
    }

private:
    T* mutex_;
    __device__ _lock() {}
};

#endif // __CUDACC__

#endif // HGCALML_MODULES_COMPILED_CUDA_DEVICE_HELPERS_H_ 