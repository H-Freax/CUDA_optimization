# CUDA优化入门实战



注：本markdown代码部分格式高亮基于c

## 新建一个CUDA项目（first_cuda.cu）

首先需要引用所需的库

要使用到runtime API的时候需要引用cuda_runtime.h

```c
#include <stdio.h>
#include <cuda_runtime.h>
```

编写主函数部分

判断CUDA是否可用

```c
int main(){
    if(!InitCUDA()){
        return 0;
    }
    
    printf("CUDA initialized.\n");

    return 0;
}
```

判断CUDA是否可用

```c
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

```c
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

#define DATA_SIZE 1048576

int data[DATA_SIZE];
```

编写生成随机数的函数

```c
void GenerateNumbers(int *number, int size){
    for(int i = 0; i < size; i++){
        number[i] = rand() % 10;
    }
}
```

CUDA执行计算前需要把数据从主记忆体复制到显示记忆体中，才能让显示晶片使用，因此需要获取一块大小适当的显示记忆体来复制数据。在main函数中加入以下部分

```c
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

```c
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

```c
函数名称<<<block 数目, thread 数目, shared memory 大小>>>(参数);
```

执行完毕后需要将结果从显示晶片复制回主记忆体，在main函数中加入以下部分

```c
    sumOfSquares<<<1, 1, 0>>>(gpudata, result);

    int sum;
    cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(gpudata);
    cudaFree(result);

    printf("sum: %d\n", sum);
```

编写的部分设置了只适用一个thread，所以block数目跟thread数目都为1，没有使用到shared memory，设为0。

如果要验证结果是否正确，可以加入CPU的执行代码进行验证

```c
    sum = 0;
    for(int i = 0; i < DATA_SIZE; i++) {
        sum += data[i] * data[i];
    }
    printf("CPU sum: %d\n", sum);
```

CUDA提供了clock函数可以取得timestamp，适合用来判断程序执行的时间，单位为GPU执行单元的时间，可以在优化程序时间时提供参考，如果需要记录时间的话，需要更改sumOfSquares为以下内容

```c
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

```c
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

```c
#define DATA_SIZE    1048576
#define THREAD_NUM   256
```

修改sumOfSquares为以下内容，即__优化1.0__

```c
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

```c
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

```c
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

```c
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

```c
#define DATA_SIZE   1048576
#define BLOCK_NUM   32
#define THREAD_NUM   256
```

表示接下来我们会用到32个block，每个block有256个threads，一共有32*256=8192个threads

接下来修改sumOfSquares部分

```c
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

```c
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

```c
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

```c
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

```c
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

```c
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

```c
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

### 先来试试基本的矩阵乘法（1.0）

简单起见，我们这里先采用方形矩阵举例：

假设有两个矩阵 A跟 B，计算 AB=C 可以用以下代码表示

```c
for(i = 0; i < n; i++) {
	for(j = 0; j < n; j++) {
			C[i][j] = 0;
			for(k = 0; k < n; k++) {
					C[i][j] += A[i][k] * B[k][j];
      }
  }
}
```

然后我们完善一下 main 函数

```c
int main(){
    float *a, *b, *c, *d;
    int n = 1000;

    if(!InitCUDA()) return 0;

    a = (float*) malloc(sizeof(float) * n * n);
    b = (float*) malloc(sizeof(float) * n * n);
    c = (float*) malloc(sizeof(float) * n * n);
    d = (float*) malloc(sizeof(float) * n * n);

    srand(0);

    matgen(a, n, n);
    matgen(b, n, n);

    clock_t time = matmultCUDA(a, n, b, n, c, n, n);

    matmult(a, n, b, n, d, n, n);
    compare_mat(c, n, d, n, n);

    double sec = (double) time / CLOCKS_PER_SEC;
    printf("Time used: %.2f (%.2lf GFLOPS)\n", sec,
       2.0 * n * n * n / (sec * 1E9));

    return 0;    
}

```

接下来编写用于产生矩阵的matgen 函数

```c
void matgen(float* a, int lda, int n){
    int i, j;

    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            a[i * lda + j] = (float) rand() / RAND_MAX + 
                (float) rand() / (RAND_MAX * RAND_MAX);
        }
    }
}
```

先用将矩阵填满0~1之间的随机数，由于C语言中不能给出可以变动的二维矩阵，因此采用i * lda + j的方式来声明。

接下来编写矩阵乘法的部分，即 matmult 函数

```c
void matmult(const float* a, int lda, const float* b, int ldb, 
    float* c, int ldc, int n){
    int i, j, k;

    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            double t = 0;
            for(k = 0; k < n; k++) {
                t += a[i * lda + k] * b[k * ldb + j];
            }
            c[i * ldc + j] = t;
        }
    }
}
```

这个部分采用 CPU 进行计算，用于比较答案是否正确，采用 double 格式进行结果的保存提高精度。

接下来我们来编写用于验证结果是否正确的函数 compare_mat

```c
void compare_mat(const float* a, int lda, 
    const float* b, int ldb, int n){
    float max_err = 0;
    float average_err = 0;
    int i, j;
    
    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            if(b[i * ldb + j] != 0) {
                float err = fabs((a[i * lda + j] -
                    b[i * ldb + j]) / b[i * ldb + j]);
                if(max_err < err) max_err = err;
                average_err += err;
            }
        }
    }

    printf("Max error: %g Average error: %g\n",
        max_err, average_err / (n * n));
}
```

计算使用 CPU 计算的矩阵结果跟使用 GPU计算的矩阵结果之间的最大误差以及平均误差。

最后编写使用 CUDA 计算矩阵的部分

```c
 #define NUM_THREADS 256

clock_t matmultCUDA(const float* a, int lda,
		const float* b, int ldb, float* c, int ldc, int n)
		{
				float *ac, *bc, *cc;
        clock_t start, end;

        start = clock();
        cudaMalloc((void**) &ac, sizeof(float) * n * n);
        cudaMalloc((void**) &bc, sizeof(float) * n * n);
        cudaMalloc((void**) &cc, sizeof(float) * n * n);

        cudaMemcpy2D(ac, sizeof(float) * n, a, sizeof(float) * lda,
            sizeof(float) * n, n, cudaMemcpyHostToDevice);
        cudaMemcpy2D(bc, sizeof(float) * n, b, sizeof(float) * ldb,
            sizeof(float) * n, n, cudaMemcpyHostToDevice);

        int blocks = (n + NUM_THREADS - 1) / NUM_THREADS;
        matMultCUDA<<<blocks * n, NUM_THREADS>>>
            (ac, n, bc, n, cc, n, n);

        cudaMemcpy2D(c, sizeof(float) * ldc, cc, sizeof(float) * n,
        sizeof(float) * n, n, cudaMemcpyDeviceToHost);

        cudaFree(ac);
        cudaFree(bc);
        cudaFree(cc);

        end = clock();

        return end - start;
}
```

在这个计算过程中，因为使用 cudaMemcpy 函数来进行复制的话，对于二维矩阵而言需要按行分开复制，降低效率，因此采用了 cudaMemcpy2D函数用来复制。

该函数中进行计算的 kernel 如下

```c
__global__ static void matMultCUDA(const float* a, size_t lda,
    const float* b, size_t ldb, float* c, size_t ldc, int n){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int row = idx / n;
    const int column = idx % n;
    int i;

    if(row < n && column < n) {
        float t = 0;
        for(i = 0; i < n; i++) {
            t += a[row * lda + i] * b[i * ldb + column];
        }
        c[row * ldc + column] = t;
    }
    
}
```

一开始先从 bid 和 tid 计算出该thread 应该计算的行跟列，在判断行跟列在范围内以后就可以进行计算，然后写入 C 矩阵中。

但是这样的计算方式执行效率较低，产生的误差也偏高。误差大的原因是我们在 CPU 上进行计算时采用的是 double 格式进行累计，在GPU 上采用的是 float 格式进行累计，累计大量数据以后误差也会变大。因此我们可以采用Kahan's Summation Formula来提高精度。

即采用一个中间变量保存有可能被吞掉的部分。具体实现如下

```c
if(row < n && column < n) {
        float t = 0;
        float y = 0;
        for(i = 0; i < n; i++) {
            float r;
            y -= a[row * lda + i] * b[i * ldb + column];
            r = t - y;
            y = (r - t) + y;
            t = r;
        }
    }
```

 修改后误差虽然减小了，但是会存在效率没有什么变化的问题。

这是因为采用Kahan's Summation Formula需要计算的量会增大，主要的瓶颈应该是在存取上，大量的读取过程是重复的，例如矩阵的每行在每次计算时都需要被重复读入，需要读取2*n^3次，如果每个行只用读一次的话，会减少到 n^3+n^2次。

### 来尝试一下第一次改良把！（改良2.0）

我们可以采用 shared memory来存储每行的内容，因为同一个 block 的 threads才能共用 shared memory，因此一行只能由同一个 block 的 threads 来计算，另外也需要存下一整行的 shared memory，所以需要把 kernel 的部分修改如下

```c
matMultCUDA<<<n, NUM_THREADS, sizeof(float) * n>>>
            (ac, n, bc, n, cc, n, n);
```

kernel 的部分修改如下

```c
    __global__ static void matMultCUDA(const float* a, size_t lda,
        const float* b, size_t ldb, float* c, size_t ldc, int n)
    {
        extern __shared__ float data[];
        const int tid = threadIdx.x;
        const int row = blockIdx.x;
        int i, j;

        for(i = tid; i < n; i += blockDim.x) {
            data[i] = a[row * lda + i];
        }

        __syncthreads();

        for(j = tid; j < n; j += blockDim.x) {
            float t = 0;
            float y = 0;
            for(i = 0; i < n; i++) {
                float r;
                y -= data[i] * b[i * ldb + j];
                r = t - y;
                y = (r - t) + y;
                t = r;
            }
            c[row * ldc + j] = t;
        }
    }
```

虽然速度有一点提高，但是还有提升空间，比如说虽然 A 矩阵的行不用重复读取了，但是 B 矩阵的列还需要被重复读取。

还有一个看起来不明显的问题，对 B 矩阵的读取虽然看起来不连续，但是实际上的连续的，因为不同的 thread 读取不同的列，因此某个时间里每个 thread读取的各个列加起来就是一个连续的记忆体区块。之所以效率没有那么好，是因为 GPU 上的记忆体控制器需要按某个固定的倍数地址开始读取才会效率最高，例如16bytes 的倍数，因此，矩阵的大小即行数列数并不是16的倍数，所以效率不佳。

想解决这个问题的话，可以在 cudaMalloc 的地方稍微适当修改，让尺寸变成适当的倍数就可以。在 CUDA 中，可以采用cudaMallocPitch函数，可以自动匹配最佳的倍数，因此可以把 cudaMalloc 的部分改成

```c
		size_t pitch_a, pitch_b, pitch_c;
    cudaMallocPitch((void**) &ac, &pitch_a, sizeof(float) * n, n);
    cudaMallocPitch((void**) &bc, &pitch_b, sizeof(float) * n, n);
    cudaMallocPitch((void**) &cc, &pitch_c, sizeof(float) * n, n);
```

同时需要在复制阶段把宽度换成对应的

```c
cudaMemcpy2D(ac, pitch_a, a, sizeof(float) * lda,
        sizeof(float) * n, n, cudaMemcpyHostToDevice);
    cudaMemcpy2D(bc, pitch_b, b, sizeof(float) * ldb,
        sizeof(float) * n, n, cudaMemcpyHostToDevice);
```

kernel的部分也需要做对应的更改

```c
    matMultCUDA<<<n, NUM_THREADS, sizeof(float) * n>>>
        (ac, pitch_a / sizeof(float), bc, pitch_b / sizeof(float),
        cc, pitch_c / sizeof(float), n);
```

 复制回主记忆体的部分也需要修改

```c
    cudaMemcpy2D(c, sizeof(float) * ldc, cc, pitch_c,
        sizeof(float) * n, n, cudaMemcpyDeviceToHost);
```

经过这样优化以后，速度可以提升三倍多。

虽然执行速度提高了很多，但是可以看出，这个改良受限于记忆体的频宽。

### 可以试试不被记忆体的频宽限制进行改良吗？（改良3.0）

上一节中提到过 B 矩阵的列也被重复使用，理论上应该也可以避免重复载入，但是因为 B 矩阵的列的使用时间跟 A 矩阵的行不一样，所以不能直接按照 A 矩阵的方法来做。

解决的方法是采用 blocking，也就是把大矩阵拆成很多小矩阵来计算，例如如果我们需要计算 C 矩阵中[0,0]-[15,15]的值，可以看做

A[0~15, 0~15] * B[0~15, 0~15] +A[0~15, 16~31] * B[16~31, 0~15] +...

这样一来，我们只需要把小矩阵载入 shared memory，小矩阵的计算就不需要外部记忆体了，这样来说，假设小矩阵的记忆体大小是 k，实际上需要的记忆体存储次数就是
$$
2k^2(n/k)^3=2n^3/k
$$
为了方便block的计算，我们让每个 block 拥有16*16个 threads，再建立（n/16)*(n/16)个 blocks

我们修改一下以下部分

```c
    int bx = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blocks(bx, bx);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    matMultCUDA<<<blocks, threads>>>(ac, pitch_a / sizeof(float),
        bc, pitch_b / sizeof(float), cc, pitch_c / sizeof(float), n);
```

BLOCK_SIZE定义成16，dim3在 CUDA 中表示一个3D 的向量。

kernel 的部分修改如下

```c
    __global__ static void matMultCUDA(const float* a, size_t lda,
        const float* b, size_t ldb, float* c, size_t ldc, int n)
    {
        __shared__ float matA[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float matB[BLOCK_SIZE][BLOCK_SIZE];
        const int tidc = threadIdx.x;
        const int tidr = threadIdx.y;
        const int bidc = blockIdx.x * BLOCK_SIZE;
        const int bidr = blockIdx.y * BLOCK_SIZE;
        int i, j;

        float results = 0;
        float comp = 0;

        for(j = 0; j < n; j += BLOCK_SIZE) {
            if(tidr + bidr < n && tidc + j < n) {
                matA[tidr][tidc] = a[(tidr + bidr) * lda + tidc + j];
            }
            else {
                matA[tidr][tidc] = 0;
            }

            if(tidr + j < n && tidc + bidc < n) {
                matB[tidr][tidc] = b[(tidr + j) * ldb + tidc + bidc];
            }
            else {
                matB[tidr][tidc] = 0;
            }

            __syncthreads();

            for(i = 0; i < BLOCK_SIZE; i++) {
                float t;
                comp -= matA[tidr][i] * matB[i][tidc];
                t = results - comp;
                comp = (t - results) + comp;
                results = t;
            }

            __syncthreads();
        }

        if(tidr + bidr < n && tidc + bidc < n) {
            c[(tidr + bidr) * ldc + tidc + bidc] = results;
        }
    }
```

因为16*16的 threads，threadIdx 可以取 threadIdx.x和 threadIdx.y，范围分别为0~15，blockIdx.x 和 blockIdx.y 也是一样，范围分别为0~n/16.

因为矩阵大小不一定是16的倍数，因此需要使用 if 进行判断。

但是这样改进以后，提升的较少，效率并没有提升到想象中的八倍。让我们想想还有什么改进空间？

### 还有什么改进空间吗？（改良4.0）

在上一节中，因为存在大量的 if 来判断，可以考虑是不是在配置的时候就配置成16的倍数，并且复制到记忆体之前先清零

```c
    int newn = ((n + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

    cudaMallocPitch((void**) &ac, &pitch_a,
        sizeof(float) * newn, newn);
    cudaMallocPitch((void**) &bc, &pitch_b,
        sizeof(float) * newn, newn);
    cudaMallocPitch((void**) &cc, &pitch_c,
        sizeof(float) * newn, newn);

    cudaMemset(ac, 0, pitch_a * newn);
    cudaMemset(bc, 0, pitch_b * newn);
```

 这样我们就可以移除所有的 if 判断了

```c
    __global__ static void matMultCUDA(const float* a, size_t lda,
        const float* b, size_t ldb, float* c, size_t ldc, int n)
    {
        __shared__ float matA[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float matB[BLOCK_SIZE][BLOCK_SIZE];
        const int tidc = threadIdx.x;
        const int tidr = threadIdx.y;
        const int bidc = blockIdx.x * BLOCK_SIZE;
        const int bidr = blockIdx.y * BLOCK_SIZE;
        int i, j;

        float results = 0;
        float comp = 0;

        for(j = 0; j < n; j += BLOCK_SIZE) {
            matA[tidr][tidc] = a[(tidr + bidr) * lda + tidc + j];
            matB[tidr][tidc] = b[(tidr + j) * ldb + tidc + bidc];

            __syncthreads();

            for(i = 0; i < BLOCK_SIZE; i++) {
                float t;
                comp -= matA[tidr][i] * matB[i][tidc];
                t = results - comp;
                comp = (t - results) + comp;
                results = t;
            }

            __syncthreads();
        }

        c[(tidr + bidr) * ldc + tidc + bidc] = results;
    }
```

