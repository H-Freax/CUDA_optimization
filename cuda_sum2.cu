#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

#define DATA_SIZE 1048576
//加入 block 共享内存
#define BLOCK_NUM   32
//加入线程
#define THREAD_NUM 256


int data[DATA_SIZE];



//初始化CUDA
bool InitCUDA(){
    int count;
    
    cudaGetDeviceCount(&count);
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

//创建0-9的随机数
void GenerateNumbers(int *number, int size){
    for(int i = 0; i < size; i++){
        number[i] = rand() % 10;
    }
}

//显示晶片上执行【未优化】
// __global__ static void sumOfSquares(int *num, int* result, clock_t* time){
//     int sum = 0;
//     int i;
//     clock_t start = clock();
//     for(i =0; i<DATA_SIZE; i++){
//         sum+= num[i]*num[i];
//     }

//     *result = sum;
//     *time = clock()-start;
// }

//优化1.0
// __global__ static void sumOfSquares(int *num, int* result,clock_t* time){
//     const int tid = threadIdx.x;
//     const int size = DATA_SIZE / THREAD_NUM;
//     int sum = 0;
//     int i;
//     clock_t start;
//     if(tid == 0) start = clock();
//     for(i = tid * size; i < (tid + 1) * size; i++){
//         sum += num[i] * num[i];
//     }

//     result[tid] = sum;
//     if(tid == 0) *time = clock()- start;
// }

//优化2.0
// __global__ static void sumOfSquares(int *num, int* result, clock_t* time)
// {
//     const int tid = threadIdx.x;
//     int sum = 0;
//     int i;
//     clock_t start;
//     if(tid == 0) start = clock();
//     for(i = tid; i < DATA_SIZE; i += THREAD_NUM) {
//        sum += num[i] * num[i];
//     }

//     result[tid] = sum;
//     if(tid == 0) *time = clock() - start;
// }

//优化3.0
// __global__ static void sumOfSquares(int *num, int* result, clock_t* time)
// {
//     const int tid = threadIdx.x;
//     const int bid = blockIdx.x;
//     int sum = 0;
//     int i;
//     if(tid == 0) time[bid] = clock();
//     for(i = bid * THREAD_NUM + tid; i < DATA_SIZE;
//         i += BLOCK_NUM * THREAD_NUM) {
//        sum += num[i] * num[i];
//     }

//     result[bid * THREAD_NUM + tid] = sum;
//     if(tid == 0) time[bid + BLOCK_NUM] = clock();
// }

//优化4.0
__global__ static void sumOfSquares(int *num, int* result, clock_t* time)
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

//优化5.0 树状加法
__global__ static void sumOfSquares(int *num, int* result, clock_t* time)
{
    extern __shared__ int shared[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int i;
    int offset = 1, mask = 1;
    if(tid == 0) time[bid] = clock();
    shared[tid] = 0;

    //树状加法展开
    // if(tid < 128) { shared[tid] += shared[tid + 128]; }
    // __syncthreads();
    // if(tid < 64) { shared[tid] += shared[tid + 64]; }
    // __syncthreads();
    // if(tid < 32) { shared[tid] += shared[tid + 32]; }
    // __syncthreads();
    // if(tid < 16) { shared[tid] += shared[tid + 16]; }
    // __syncthreads();
    // if(tid < 8) { shared[tid] += shared[tid + 8]; }
    // __syncthreads();
    // if(tid < 4) { shared[tid] += shared[tid + 4]; }
    // __syncthreads();
    // if(tid < 2) { shared[tid] += shared[tid + 2]; }
    // __syncthreads();
    // if(tid < 1) { shared[tid] += shared[tid + 1]; }
    // __syncthreads();
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

//树状加法防止 share memory 的 bank conflict 的问题
    // offset = THREAD_NUM / 2;
    // while(offset > 0) {
    //     if(tid < offset) {
    //         shared[tid] += shared[tid + offset];
    //     }
    //     offset >>= 1;
    //     __syncthreads();
    // }


int main(){
    if(!InitCUDA()){
        return 0;
    }
    
    printf("CUDA initialized.\n");

    GenerateNumbers(data,DATASIZE);


    /*
        优化2.0 & 1.0
    */
    // int* gpudata, *result;
    // clock_t* time;
    // cudaMalloc((void**) &gpudata,sizeof(int) * DATA_SIZE);
    // //扩大 result
    // cudaMalloc((void**) &result,sizeof(int) * THREAD_NUM);
    // cudaMalloc((void**) &time, sizeof(clock_t));
    // //从主记忆体复制到显示记忆体，所以使用 cudaMemcpyHostToDevice。
    // //如果是从显示记忆体复制到主记忆体，则使用 cudaMemcpyDeviceToHost
    // cudaMemcpy(gpudata, data,sizeof(int) * DATA_SIZE,
    //     cudaMemcpyHostToDevice);

    // //执行函数语法：
    // //函数名称<<<block 数目, thread 数目, shared memory 大小>>>(参数);
    // // sumOfSquares<<<1, 1, 0>>>(gpudata,result,time);
    // sumOfSquares<<<1, THREAD_NUM, 0>>>(gpudata,result,time);

    // // int sum;
    // int sum[THREAD_NUM];
    // clock_t time_used;
    // // cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&sum, result, sizeof(int) * THREAD_NUM, cudaMemcpyDeviceToHost);
    // cudaMemcpy(&time_used, time, sizeof(clock_t), cudaMemcpyDeviceToHost);
    // cudaFree(gpudata);
    // cudaFree(result);
    // cudaFree(time)

    // // printf("sum: %d time: %d\n", sum, time_used);
    // //CPU端统计
    // int final_sum = 0;
    // for(int i = 0; i < THREAD_NUM; i++) {
    //     final_sum += sum[i];
    // }

    // printf("sum: %d  time: %d\n", final_sum, time_used);

    // final_sum = 0;
    // for(int i = 0; i < DATA_SIZE; i++) {
    //     sum += data[i] * data[i];
    // }
    // printf("sum (CPU): %d\n", final_sum);



    /*
        优化3.0
    */

    // int* gpudata, *result;
    // clock_t* time;
    // cudaMalloc((void**) &gpudata, sizeof(int) * DATA_SIZE);
    // cudaMalloc((void**) &result, sizeof(int) * THREAD_NUM * BLOCK_NUM);
    // cudaMalloc((void**) &time, sizeof(clock_t) * BLOCK_NUM * 2);
    // cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE,
    //     cudaMemcpyHostToDevice);

    // sumOfSquares<<<BLOCK_NUM, THREAD_NUM, 0>>>(gpudata, result,
    //     time);

    // int sum[THREAD_NUM * BLOCK_NUM];
    // clock_t time_used[BLOCK_NUM * 2];
    // cudaMemcpy(&sum, result, sizeof(int) * THREAD_NUM * BLOCK_NUM,
    //     cudaMemcpyDeviceToHost);
    // cudaMemcpy(&time_used, time, sizeof(clock_t) * BLOCK_NUM * 2,
    //     cudaMemcpyDeviceToHost);
    // cudaFree(gpudata);
    // cudaFree(result);
    // cudaFree(time);

    // int final_sum = 0;
    // for(int i = 0; i < THREAD_NUM * BLOCK_NUM; i++) {
    //     final_sum += sum[i];
    // }

    // clock_t min_start, max_end;
    // min_start = time_used[0];
    // max_end = time_used[BLOCK_NUM];
    // for(int i = 1; i < BLOCK_NUM; i++) {
    //     if(min_start > time_used[i])
    //         min_start = time_used[i];
    //     if(max_end < time_used[i + BLOCK_NUM])
    //         max_end = time_used[i + BLOCK_NUM];
    // }

    // printf("sum: %d  time: %d\n", final_sum, max_end - min_start);
    
    
    
    /*
        优化4.0 &5.0
    */

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


    return 0;



    
}

