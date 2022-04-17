# CUDA优化入门实战

注：本markdown代码部分格式高亮基于c++
- [CUDA优化入门实战](#cuda------)
  * [新建一个CUDA项目（first_cuda.cu）](#----cuda---first-cudacu-)
  * [用CUDA做个简单的加法(cuda_sum1.cu)](#-cuda--------cuda-sum1cu-)
  * [试试看优化CUDA上的加法吧(cuda_sum2.cu)](#-----cuda------cuda-sum2cu-)
    + [先来试试并行化（优化1.0）](#----------10-)
    + [来试试基于记忆体的存取模式的优化吧（优化2.0）](#--------------------20-)
    + [还能有更多的并行化吗？（优化3.0）](#--------------30-)
    + [来试试Thread的同步（优化4.0）](#---thread------40-)
    + [来试试用树状加法优化（优化5.0）](#-------------50-)
    + [还有什么改进空间吗？](#----------)
  * [用CUDA做个简单的乘法(cuda_float.cu)](#-cuda--------cuda-floatcu-)

## 新建一个CUDA项目（first_cuda.cu）

首先需要引用所需的库

要使用到runtime API的时候需要引用cuda_runtime.h

```cmake
#include <stdio.h>
#include <cuda_runtime.h>
```

编写主函数部分

判断CUDA是否可用

```cmake
int main(){
    if(!InitCUDA()){
        return 0;
    }
    
    printf("CUDA initialized.\n");

    return 0;
}
```

判断CUDA是否可用

```c++
bool InitCUDA(){
    int count;
    
    cudaGetDeviceCount(&count); //获取可用的device数目
    if(count == 0){
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    
    int i;	
    for(int i = 0; i<count;i++){
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess){
            if(prop.major >= 1){
                break;
            }
        }
    }
    
    if(i == count){
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }

    cudaSetDevice(i);

    return true;
}
```

首先使用cudaGetDeviceCount函数获取支持CUDA的device数目，如果没有的话会返回1，device 0 会是一个模拟的device但是不支持CUDA 1.0以上的功能。

需要对每个device使用cudaGetDeviceProperties函数取得device的相关资料，例如device的名称、memory的大小，最大的thread数目等等。同时通过该函数判断该device所支持的CUDA版本（prop.majar和prop.minor分别代表装备支持的版本号，例如1.0即prop.major =1，prop.minor为0

如果找到了支持CUDA 1.0版本以上的device以后，可以使用cudaSetDevice将该device设为目前需要使用的device



完成以上内容以后就可以使用nvcc来complie这个文件了。

- nvcc是CUDA的compile工具，会将.cu拆解出GPU或者host上需要执行的部分。在GPU上执行的部分会采用NVIDIA提供的compiler编译成中介码，host上执行的部分会通过系统上的C++ compiler编译。

## 用CUDA做个简单的加法(cuda_sum1.cu)

该部分在first_cuda.cu的基础上进行

目标是计算一堆数字的平方和

首先把开头部分改成

```c++
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

#define DATA_SIZE 1048576

int data[DATA_SIZE];
```

编写生成随机数的函数

```c++
void GenerateNumbers(int *number, int size){
    for(int i = 0; i < size; i++){
        number[i] = rand() % 10;
    }
}
```

CUDA执行计算前需要把数据从主记忆体复制到显示记忆体中，才能让显示晶片使用，因此需要获取一块大小适当的显示记忆体来复制数据。在main函数中加入以下部分

```c++
   GenerateNumbers(data,DATASIZE);
    int* gpudata, *result;

    cudaMalloc((void**) &gpudata,sizeof(int) * DATA_SIZE);
    cudaMalloc((void**) &result,sizeof(int));
    //从主记忆体复制到显示记忆体，所以使用 cudaMemcpyHostToDevice。
    //如果是从显示记忆体复制到主记忆体，则使用 cudaMemcpyDeviceToHost
    cudaMemcpy(gpudata, data,sizeof(int) * DATA_SIZE,
        cudaMemcpyHostToDevice);
```

首先引用GenerateNumbers函数产生数据，使用cudaMalloc取得大小合适的记忆体，gpudata变量用来存储数据，result用来存储计算结果，通过cudaMemcpy将数据复制到显示记忆体中。

cudaMalloc和cudaMemcpy的用法和一般的malloc以及memcpy类似，不过cudaMemcpy多出一个参数，指示复制记忆体的方向，如果是从主记忆体复制到显示记忆体，使用 cudaMemcpyHostToDevice，如果是从显示记忆体复制到主记忆体，则使用 cudaMemcpyDeviceToHost。

接下来编写在显示晶片上的函数，在CUDA中，在函数前面加上\__global__表示这个函数是在显示晶片上执行的。

```c++
__global__ static void sumOfSquares(int *num, int* result)
{
    int sum = 0;
    int i;
    for(i = 0; i < DATA_SIZE; i++) {
        sum += num[i] * num[i];
    }

    *result = sum;
}
```

在显示晶片上执行程序有一定的限制条件，例如不能传回值。

接下来是让CUDA执行函数，在CUDA中执行函数需要使用以下的语法

```c++
函数名称<<<block 数目, thread 数目, shared memory 大小>>>(参数);
```

执行完毕后需要将结果从显示晶片复制回主记忆体，在main函数中加入以下部分

```c++
    sumOfSquares<<<1, 1, 0>>>(gpudata, result);

    int sum;
    cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(gpudata);
    cudaFree(result);

    printf("sum: %d\n", sum);
```

编写的部分设置了只适用一个thread，所以block数目跟thread数目都为1，没有使用到shared memory，设为0。

如果要验证结果是否正确，可以加入CPU的执行代码进行验证

```c++
    sum = 0;
    for(int i = 0; i < DATA_SIZE; i++) {
        sum += data[i] * data[i];
    }
    printf("CPU sum: %d\n", sum);
```

CUDA提供了clock函数可以取得timestamp，适合用来判断程序执行的时间，单位为GPU执行单元的时间，可以在优化程序时间时提供参考，如果需要记录时间的话，需要更改sumOfSquares为以下内容

```c++
__global__ static void sumOfSquares(int *num, int* result, clock_t* time){
    int sum = 0;
    int i;
    clock_t start = clock();
    for(i =0; i<DATA_SIZE; i++){
        sum+= num[i]*num[i];
    }

    *result = sum;
    *time = clock()-start;
}
```

main函数的部分修改如下：

```c++
    int* gpudata, *result;
    clock_t* time;
    cudaMalloc((void**) &gpudata, sizeof(int) * DATA_SIZE);
    cudaMalloc((void**) &result, sizeof(int));
    cudaMalloc((void**) &time, sizeof(clock_t));
    cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE,
        cudaMemcpyHostToDevice);

    sumOfSquares<<<1, 1, 0>>>(gpudata, result, time);

    int sum;
    clock_t time_used;
    cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&time_used, time, sizeof(clock_t),
        cudaMemcpyDeviceToHost);
    cudaFree(gpudata);
    cudaFree(result);

    printf("sum: %d time: %d\n", sum, time_used);

```

编译后运行就可以查看所花费的时间了。

## 试试看优化CUDA上的加法吧(cuda_sum2.cu)

### 先来试试并行化（优化1.0）

在cuda_sum1.cu中，并没有进行并行化，整个程序只有一个thread，效果并不是很好。

这主要是因为GPU的架构导致的，在CUDA中，一般的内容复制到显示记忆体中的部分，即global memory，这些部分是没有cache的，而且存取global memory的时间较长，通常是数百个cycle。由于程序只有一个thread，每次读取global memory 的内容需要等到读取到内容、累加以后才能进行下一步。

由于global memory没有cache，如果想避开存取的巨量时间，就需要利用大量threads。

要如何把计算平方和的程序并行化呢？我们可以把数字分成若干组，分别计算平方和以后进行相加。

一开始，我们可以把最后的和的相加放在CPU上执行。

首先，基于first_cuda.cu，添加以下代码

```c++
#define DATA_SIZE    1048576
#define THREAD_NUM   256
```

修改sumOfSquares为以下内容，即__优化1.0__

```c++
__global__ static void sumOfSquares(int *num, int* result,
    clock_t* time)
{
    const int tid = threadIdx.x;
    const int size = DATA_SIZE / THREAD_NUM;
    int sum = 0;
    int i;
    clock_t start;
    if(tid == 0) start = clock();
    for(i = tid * size; i < (tid + 1) * size; i++) {
       sum += num[i] * num[i];
    }

    result[tid] = sum;
    if(tid == 0) *time = clock() - start;
}
```

其中，threadIdx.x是CUDA中表示目前的thread是第几个thread的变量，该变量从0开始计算，由于我们设置有256个thread，所以计数为0~255，利用这个变量，我们可以计算各个组的平方和，另外当threadIdx.x=0的时候可以进行计算时间的start。

因为会有256个计算结果，所以result也需要扩大占用的位置。需要修改main函数为

```c++
    int* gpudata, *result;
    clock_t* time;
    cudaMalloc((void**) &gpudata, sizeof(int) * DATA_SIZE);
    cudaMalloc((void**) &result, sizeof(int) * THREAD_NUM);
    cudaMalloc((void**) &time, sizeof(clock_t));
    cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE,
        cudaMemcpyHostToDevice);

    sumOfSquares<<<1, THREAD_NUM, 0>>>(gpudata, result, time);

    int sum[THREAD_NUM];
    clock_t time_used;
    cudaMemcpy(&sum, result, sizeof(int) * THREAD_NUM,
        cudaMemcpyDeviceToHost);
    cudaMemcpy(&time_used, time, sizeof(clock_t),
        cudaMemcpyDeviceToHost);
    cudaFree(gpudata);
    cudaFree(result);
    cudaFree(time);
```

最后，在CPU端计算和

```c++
    int final_sum = 0;
    for(int i = 0; i < THREAD_NUM; i++) {
        final_sum += sum[i];
    }

    printf("sum: %d  time: %d\n", final_sum, time_used);

    final_sum = 0;
    for(int i = 0; i < DATA_SIZE; i++) {
        sum += data[i] * data[i];
    }
    printf("sum (CPU): %d\n", final_sum);
```

编译后可以发现结果相同的前提下，速度快了77倍！

### 来试试基于记忆体的存取模式的优化吧（优化2.0）

显示卡上的记忆体是DRAM（即动态随机存取存储器，主要的作用原理是利用电容内存储电荷的多寡来代表一个二进制比特是1还是0。）因此最有效率的存储方式，是连续进行存储。

上面编写的程序看是连续存储记忆体的位置，但是考虑到thread的执行方式，当一个thread在等待内容的时候，GPU会切换到下一个thread。也就是说实际上执行的顺序是类似

thread 0 -> thread 1 -> thread 2 -> ……

因此，在同一个thread中连续存储记忆体，实际执行的时候并不连续，要让实际执行时连续，应该要让thread 0读取第一个数，thread 1读取第二个数，以此类推。因此需要修改sumOfSquares如下

```c++
__global__ static void sumOfSquares(int *num, int* result,
    clock_t* time)
{
    const int tid = threadIdx.x;
    int sum = 0;
    int i;
    clock_t start;
    if(tid == 0) start = clock();
    for(i = tid; i < DATA_SIZE; i += THREAD_NUM) {
       sum += num[i] * num[i];
    }

    result[tid] = sum;
    if(tid == 0) *time = clock() - start;
}
```

编译后执行结果相同，又比上一版快了三倍！

如果增加thread数目的数目，就可以看到更好的效率，例如512个，主要取决于GPU的block中最多能有几个thread，同时如果thread数目增加太多，CPU端相加的工作也会变多。

### 还能有更多的并行化吗？（优化3.0）

上面提到了block，接下来我们来介绍一下block。

在CUDA中，thread可以进行分组，也就是block，一个block中的thread有一个共用的shared memory，可以进行同步工作。不同的block之间的thread不行。接下来我们试试用多个block来进一步增加thread的数目。

首先在#define的位置修改代码

```c++
#define DATA_SIZE   1048576
#define BLOCK_NUM   32
#define THREAD_NUM   256
```

表示接下来我们会用到32个block，每个block有256个threads，一共有32*256=8192个threads

接下来修改sumOfSquares部分

```c++
__global__ static void sumOfSquares(int *num, int* result,
    clock_t* time)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int sum = 0;
    int i;
    if(tid == 0) time[bid] = clock();
    for(i = bid * THREAD_NUM + tid; i < DATA_SIZE;
        i += BLOCK_NUM * THREAD_NUM) {
       sum += num[i] * num[i];
    }

    result[bid * THREAD_NUM + tid] = sum;
    if(tid == 0) time[bid + BLOCK_NUM] = clock();
}
```

 blockIdx.x的用法跟threadIdx.x相同，表示的是block的编号，在这个版本中我们记录每个block的开始以及结束时间。

最后修改main函数部分

```c++
    int* gpudata, *result;
    clock_t* time;
    cudaMalloc((void**) &gpudata, sizeof(int) * DATA_SIZE);
    cudaMalloc((void**) &result,
        sizeof(int) * THREAD_NUM * BLOCK_NUM);
    cudaMalloc((void**) &time, sizeof(clock_t) * BLOCK_NUM * 2);
    cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE,
        cudaMemcpyHostToDevice);

    sumOfSquares<<<BLOCK_NUM, THREAD_NUM, 0>>>(gpudata, result,
        time);

    int sum[THREAD_NUM * BLOCK_NUM];
    clock_t time_used[BLOCK_NUM * 2];
    cudaMemcpy(&sum, result, sizeof(int) * THREAD_NUM * BLOCK_NUM,
        cudaMemcpyDeviceToHost);
    cudaMemcpy(&time_used, time, sizeof(clock_t) * BLOCK_NUM * 2,
        cudaMemcpyDeviceToHost);
    cudaFree(gpudata);
    cudaFree(result);
    cudaFree(time);

    int final_sum = 0;
    for(int i = 0; i < THREAD_NUM * BLOCK_NUM; i++) {
        final_sum += sum[i];
    }

    clock_t min_start, max_end;
    min_start = time_used[0];
    max_end = time_used[BLOCK_NUM];
    for(int i = 1; i < BLOCK_NUM; i++) {
        if(min_start > time_used[i])
            min_start = time_used[i];
        if(max_end < time_used[i + BLOCK_NUM])
            max_end = time_used[i + BLOCK_NUM];
    }

    printf("sum: %d  time: %d\n", final_sum, max_end - min_start);
```

基本上就是增加result的大小，修改计算时间的方法，把每个block最早的开始时间减去最晚的结束时间，得到最终的时间。相较上版本又快了很多，但是在CPU上的部分时间增加了，因为CPU上需要加的数字更多了，为了避免这个问题，我们可以让每个block都计算自己的threads的计算结果的和。

### 来试试Thread的同步（优化4.0）

这个版本中，我们让每个block都计算自己的threads的计算结果的和，把sumOfSquares修改如下

```c++
__global__ static void sumOfSquares(int *num, int* result,
    clock_t* time)
{
    extern __shared__ int shared[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int i;
    if(tid == 0) time[bid] = clock();
    shared[tid] = 0;
    for(i = bid * THREAD_NUM + tid; i < DATA_SIZE;
        i += BLOCK_NUM * THREAD_NUM) {
       shared[tid] += num[i] * num[i];
    }

    __syncthreads();
    if(tid == 0) {
        for(i = 1; i < THREAD_NUM; i++) {
            shared[0] += shared[i];
        }
        result[bid] = shared[0];
    }

    if(tid == 0) time[bid + BLOCK_NUM] = clock();
}
```

利用\__shared__表示这个变量存在于shared memory，是一个block中每个thread都公用的记忆体，会使用GPU上的记忆体，可以不用担心存取时间的问题。

\__syncthreads()是一个CUDA内置的函数，表示把block中的所有thread都同步到这个点再执行。

接下来把main函数部分改成

```c++
    int* gpudata, *result;
    clock_t* time;
    cudaMalloc((void**) &gpudata, sizeof(int) * DATA_SIZE);
    cudaMalloc((void**) &result, sizeof(int) * BLOCK_NUM);
    cudaMalloc((void**) &time, sizeof(clock_t) * BLOCK_NUM * 2);
    cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE,
        cudaMemcpyHostToDevice);

    sumOfSquares<<<BLOCK_NUM, THREAD_NUM,
        THREAD_NUM * sizeof(int)>>>(gpudata, result, time);

    int sum[BLOCK_NUM];
    clock_t time_used[BLOCK_NUM * 2];
    cudaMemcpy(&sum, result, sizeof(int) * BLOCK_NUM,
        cudaMemcpyDeviceToHost);
    cudaMemcpy(&time_used, time, sizeof(clock_t) * BLOCK_NUM * 2,
        cudaMemcpyDeviceToHost);
    cudaFree(gpudata);
    cudaFree(result);
    cudaFree(time);

    int final_sum = 0;
    for(int i = 0; i < BLOCK_NUM; i++) {
        final_sum += sum[i];
    }
```

可以发现，现在CPU上只需要加32个数字就可以了，又有了优化。

但是还是有优化的空间，最终的相加的工作，被分配给每个block的thread 0来进行，并不是最有效的方法，这个相加的动作是可以进行并行化的。

### 来试试用树状加法优化（优化5.0）

树状加法即透过树型结构的启发

![image](https://user-images.githubusercontent.com/35798640/163711236-52aa0027-1546-4527-af4a-c526e71e9a3b.png)

把sumOfSquares修改如下

```c++
__global__ static void sumOfSquares(int *num, int* result,
    clock_t* time)
{
    extern __shared__ int shared[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int i;
    int offset = 1, mask = 1;
    if(tid == 0) time[bid] = clock();
    shared[tid] = 0;
    for(i = bid * THREAD_NUM + tid; i < DATA_SIZE;
        i += BLOCK_NUM * THREAD_NUM) {
       shared[tid] += num[i] * num[i];
    }

    __syncthreads();
    while(offset < THREAD_NUM) {
        if((tid & mask) == 0) {
            shared[tid] += shared[tid + offset];
        }
        offset += offset;
        mask = offset + mask;
        __syncthreads();
    }

    if(tid == 0) {
        result[bid] = shared[0];   
        time[bid + BLOCK_NUM] = clock();
    }
}
```

### 还有什么改进空间吗？

上个版本的树状加法在GPU执行的时候可能会存在share memory的bank conflict的问题。

__bank conflict是什么？：__

​		在CUDA装置中，shared memory被分成数个bank，如果同时每个thread存取不同的bank，不会存在问题，当两个或多个threads存同一个bank的时候，就会产生bank conflict

可以进行如下改写

```c++
    offset = THREAD_NUM / 2;
    while(offset > 0) {
        if(tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        offset >>= 1;
        __syncthreads();
    }
```

这时候省去了mask，也有进一步提升，当然，再优化的话，可以展开树状加法

```c++
    if(tid < 128) { shared[tid] += shared[tid + 128]; }
    __syncthreads();
    if(tid < 64) { shared[tid] += shared[tid + 64]; }
    __syncthreads();
    if(tid < 32) { shared[tid] += shared[tid + 32]; }
    __syncthreads();
    if(tid < 16) { shared[tid] += shared[tid + 16]; }
    __syncthreads();
    if(tid < 8) { shared[tid] += shared[tid + 8]; }
    __syncthreads();
    if(tid < 4) { shared[tid] += shared[tid + 4]; }
    __syncthreads();
    if(tid < 2) { shared[tid] += shared[tid + 2]; }
    __syncthreads();
    if(tid < 1) { shared[tid] += shared[tid + 1]; }
    __syncthreads();
```



## 用CUDA做个简单的乘法(cuda_float.cu)

