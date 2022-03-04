#include <stdio.h>
#include <cuda_runtime.h>

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

//产生矩阵
void matgen(float* a, int lda, int n){
    int i, j;

    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            a[i * lda + j] = (float) rand() / RAND_MAX + 
                (float) rand() / (RAND_MAX * RAND_MAX);
        }
    }
}
//矩阵乘法
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

//验证结果
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


//CUDA内计算1.0
__global__ static void matMultCUDA(const float* a, size_t lda,
    const float* b, size_t ldb, float* c, size_t ldc, int n){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int row = idx / n;
    const int column = idx % n;
    int i;

    //改善误差前
    // if(row < n && column < n) {
    //     float t = 0;
    //     for(i = 0; i < n; i++) {
    //         t += a[row * lda + i] * b[i * ldb + column];
    //     }
    //     c[row * ldc + column] = t;
    // }
    //采用Kahan's Summation Formula改善误差后
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
}

//改良2.0
//  __global__ static void matMultCUDA(const float* a, size_t lda,
//     const float* b, size_t ldb, float* c, size_t ldc, int n){
//     extern __shared__ float data[];
//     const int tid = threadIdx.x;
//     const int row = blockIdx.x;
//     int i, j;

//     for(i = tid; i < n; i += blockDim.x) {
//         data[i] = a[row * lda + i];
//     }

//     __syncthreads();

//     for(j = tid; j < n; j += blockDim.x) {
//         float t = 0;
//         float y = 0;
//         for(i = 0; i < n; i++) {
//             float r;
//             y -= data[i] * b[i * ldb + j];
//             r = t - y;
//             y = (r - t) + y;
//             t = r;
//         }
//         c[row * ldc + j] = t;
//     }
// }

//改良3.0
//  __global__ static void matMultCUDA(const float* a, size_t lda,
//         const float* b, size_t ldb, float* c, size_t ldc, int n){
//     __shared__ float matA[BLOCK_SIZE][BLOCK_SIZE];
//     __shared__ float matB[BLOCK_SIZE][BLOCK_SIZE];
//     const int tidc = threadIdx.x;
//     const int tidr = threadIdx.y;
//     const int bidc = blockIdx.x * BLOCK_SIZE;
//     const int bidr = blockIdx.y * BLOCK_SIZE;
//     int i, j;

//     float results = 0;
//     float comp = 0;

//     for(j = 0; j < n; j += BLOCK_SIZE) {
//         if(tidr + bidr < n && tidc + j < n) {
//             matA[tidr][tidc] = a[(tidr + bidr) * lda + tidc + j];
//         }
//         else {
//             matA[tidr][tidc] = 0;
//         }

//         if(tidr + j < n && tidc + bidc < n) {
//             matB[tidr][tidc] = b[(tidr + j) * ldb + tidc + bidc];
//         }
//         else {
//             matB[tidr][tidc] = 0;
//         }

//         __syncthreads();

//         for(i = 0; i < BLOCK_SIZE; i++) {
//             float t;
//             comp -= matA[tidr][i] * matB[i][tidc];
//             t = results - comp;
//             comp = (t - results) + comp;
//             results = t;
//         }

//         __syncthreads();
//     }

//     if(tidr + bidr < n && tidc + bidc < n) {
//         c[(tidr + bidr) * ldc + tidc + bidc] = results;
//     }
// }

//改良4.0 
__global__ static void matMultCUDA(const float* a, size_t lda,
        const float* b, size_t ldb, float* c, size_t ldc, int n){
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


//CUDA 矩阵乘法1.0
// #define NUM_THREADS 256

// clock_t matmultCUDA(const float* a, int lda,
//     const float* b, int ldb, float* c, int ldc, int n){
//     float *ac, *bc, *cc;
//     clock_t start, end;

//     start = clock();
//     cudaMalloc((void**) &ac, sizeof(float) * n * n);
//     cudaMalloc((void**) &bc, sizeof(float) * n * n);
//     cudaMalloc((void**) &cc, sizeof(float) * n * n);

//     cudaMemcpy2D(ac, sizeof(float) * n, a, sizeof(float) * lda,
//         sizeof(float) * n, n, cudaMemcpyHostToDevice);
//     cudaMemcpy2D(bc, sizeof(float) * n, b, sizeof(float) * ldb,
//         sizeof(float) * n, n, cudaMemcpyHostToDevice);

//     int blocks = (n + NUM_THREADS - 1) / NUM_THREADS;
    
//     //初始版本
//     // matMultCUDA<<<blocks * n, NUM_THREADS>>>
//     //     (ac, n, bc, n, cc, n, n);
    
//     //改良1.0
//     matMultCUDA<<<n, NUM_THREADS, sizeof(float) * n>>>
//             (ac, n, bc, n, cc, n, n);

//     cudaMemcpy2D(c, sizeof(float) * ldc, cc, sizeof(float) * n,
//     sizeof(float) * n, n, cudaMemcpyDeviceToHost);

//     cudaFree(ac);
//     cudaFree(bc);
//     cudaFree(cc);

//     end = clock();

//     return end - start;
// }

//CUDA 矩阵乘法2.0
// #define NUM_THREADS 256

// clock_t matmultCUDA(const float* a, int lda,
//     const float* b, int ldb, float* c, int ldc, int n){
//     float *ac, *bc, *cc;
//     clock_t start, end;

//     start = clock();
//     // cudaMalloc((void**) &ac, sizeof(float) * n * n);
//     // cudaMalloc((void**) &bc, sizeof(float) * n * n);
//     // cudaMalloc((void**) &cc, sizeof(float) * n * n);
//     //可以自动以最佳的倍数来配置记忆体
//     size_t pitch_a, pitch_b, pitch_c;
//     cudaMallocPitch((void**) &ac, &pitch_a, sizeof(float) * n, n);
//     cudaMallocPitch((void**) &bc, &pitch_b, sizeof(float) * n, n);
//     cudaMallocPitch((void**) &cc, &pitch_c, sizeof(float) * n, n);

//     // cudaMemcpy2D(ac, sizeof(float) * n, a, sizeof(float) * lda,
//     //     sizeof(float) * n, n, cudaMemcpyHostToDevice);
//     // cudaMemcpy2D(bc, sizeof(float) * n, b, sizeof(float) * ldb,
//     //     sizeof(float) * n, n, cudaMemcpyHostToDevice);
//     //cudaMallocPitch函数会以适当的倍数配置记忆体，并把配置的宽度传回
//     //因此，在把矩阵复制到显示记忆体上时，要使用它传回的宽度
//     cudaMemcpy2D(ac, pitch_a, a, sizeof(float) * lda,
//         sizeof(float) * n, n, cudaMemcpyHostToDevice);
//     cudaMemcpy2D(bc, pitch_b, b, sizeof(float) * ldb,
//         sizeof(float) * n, n, cudaMemcpyHostToDevice);


//     int blocks = (n + NUM_THREADS - 1) / NUM_THREADS;
    
//     //初始版本
//     // matMultCUDA<<<blocks * n, NUM_THREADS>>>
//     //     (ac, n, bc, n, cc, n, n);
    
//     //改良1.0
//     // matMultCUDA<<<n, NUM_THREADS, sizeof(float) * n>>>
//     //        (ac, n, bc, n, cc, n, n);


//     matMultCUDA<<<n, NUM_THREADS, sizeof(float) * n>>>
//         (ac, pitch_a / sizeof(float), bc, pitch_b / sizeof(float),
//         cc, pitch_c / sizeof(float), n);
    
//     // cudaMemcpy2D(c, sizeof(float) * ldc, cc, sizeof(float) * n,
//     // sizeof(float) * n, n, cudaMemcpyDeviceToHost);

//     cudaMemcpy2D(c, sizeof(float) * ldc, cc, pitch_c,
//         sizeof(float) * n, n, cudaMemcpyDeviceToHost);


//     cudaFree(ac);
//     cudaFree(bc);
//     cudaFree(cc);

//     end = clock();

//     return end - start;
// }

//CUDA 矩阵乘法3.0 block
// #define NUM_THREADS 256

// clock_t matmultCUDA(const float* a, int lda,
//     const float* b, int ldb, float* c, int ldc, int n){
//     float *ac, *bc, *cc;
//     clock_t start, end;

//     start = clock();
//     // cudaMalloc((void**) &ac, sizeof(float) * n * n);
//     // cudaMalloc((void**) &bc, sizeof(float) * n * n);
//     // cudaMalloc((void**) &cc, sizeof(float) * n * n);
//     //可以自动以最佳的倍数来配置记忆体
//     size_t pitch_a, pitch_b, pitch_c;
//     cudaMallocPitch((void**) &ac, &pitch_a, sizeof(float) * n, n);
//     cudaMallocPitch((void**) &bc, &pitch_b, sizeof(float) * n, n);
//     cudaMallocPitch((void**) &cc, &pitch_c, sizeof(float) * n, n);

//     // cudaMemcpy2D(ac, sizeof(float) * n, a, sizeof(float) * lda,
//     //     sizeof(float) * n, n, cudaMemcpyHostToDevice);
//     // cudaMemcpy2D(bc, sizeof(float) * n, b, sizeof(float) * ldb,
//     //     sizeof(float) * n, n, cudaMemcpyHostToDevice);
//     //cudaMallocPitch函数会以适当的倍数配置记忆体，并把配置的宽度传回
//     //因此，在把矩阵复制到显示记忆体上时，要使用它传回的宽度
//     cudaMemcpy2D(ac, pitch_a, a, sizeof(float) * lda,
//         sizeof(float) * n, n, cudaMemcpyHostToDevice);
//     cudaMemcpy2D(bc, pitch_b, b, sizeof(float) * ldb,
//         sizeof(float) * n, n, cudaMemcpyHostToDevice);


//     int blocks = (n + NUM_THREADS - 1) / NUM_THREADS;
    
//     //初始版本
//     // matMultCUDA<<<blocks * n, NUM_THREADS>>>
//     //     (ac, n, bc, n, cc, n, n);
    
//     //改良1.0
//     // matMultCUDA<<<n, NUM_THREADS, sizeof(float) * n>>>
//     //        (ac, n, bc, n, cc, n, n);

//     //改良2.0
//     // matMultCUDA<<<n, NUM_THREADS, sizeof(float) * n>>>
//     //     (ac, pitch_a / sizeof(float), bc, pitch_b / sizeof(float),
//     //     cc, pitch_c / sizeof(float), n);

//     int bx = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
//     dim3 blocks(bx, bx);
//     dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
//     matMultCUDA<<<blocks, threads>>>(ac, pitch_a / sizeof(float),
//         bc, pitch_b / sizeof(float), cc, pitch_c / sizeof(float), n);
    
//     // cudaMemcpy2D(c, sizeof(float) * ldc, cc, sizeof(float) * n,
//     // sizeof(float) * n, n, cudaMemcpyDeviceToHost);

//     cudaMemcpy2D(c, sizeof(float) * ldc, cc, pitch_c,
//         sizeof(float) * n, n, cudaMemcpyDeviceToHost);


//     cudaFree(ac);
//     cudaFree(bc);
//     cudaFree(cc);

//     end = clock();

//     return end - start;
// }

//CUDA 改良版4.0 配置好记忆体的倍数，同时清空为0
#define NUM_THREADS 256

clock_t matmultCUDA(const float* a, int lda,
    const float* b, int ldb, float* c, int ldc, int n){
    float *ac, *bc, *cc;
    clock_t start, end;

    start = clock();
    // cudaMalloc((void**) &ac, sizeof(float) * n * n);
    // cudaMalloc((void**) &bc, sizeof(float) * n * n);
    // cudaMalloc((void**) &cc, sizeof(float) * n * n);
    //可以自动以最佳的倍数来配置记忆体
    size_t pitch_a, pitch_b, pitch_c;
    //改良版3.0
    // cudaMallocPitch((void**) &ac, &pitch_a, sizeof(float) * n, n);
    // cudaMallocPitch((void**) &bc, &pitch_b, sizeof(float) * n, n);
    // cudaMallocPitch((void**) &cc, &pitch_c, sizeof(float) * n, n);
    int newn = ((n + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

    cudaMallocPitch((void**) &ac, &pitch_a,
        sizeof(float) * newn, newn);
    cudaMallocPitch((void**) &bc, &pitch_b,
        sizeof(float) * newn, newn);
    cudaMallocPitch((void**) &cc, &pitch_c,
        sizeof(float) * newn, newn);

    cudaMemset(ac, 0, pitch_a * newn);
    cudaMemset(bc, 0, pitch_b * newn);



    // cudaMemcpy2D(ac, sizeof(float) * n, a, sizeof(float) * lda,
    //     sizeof(float) * n, n, cudaMemcpyHostToDevice);
    // cudaMemcpy2D(bc, sizeof(float) * n, b, sizeof(float) * ldb,
    //     sizeof(float) * n, n, cudaMemcpyHostToDevice);
    //cudaMallocPitch函数会以适当的倍数配置记忆体，并把配置的宽度传回
    //因此，在把矩阵复制到显示记忆体上时，要使用它传回的宽度
    cudaMemcpy2D(ac, pitch_a, a, sizeof(float) * lda,
        sizeof(float) * n, n, cudaMemcpyHostToDevice);
    cudaMemcpy2D(bc, pitch_b, b, sizeof(float) * ldb,
        sizeof(float) * n, n, cudaMemcpyHostToDevice);


    int blocks = (n + NUM_THREADS - 1) / NUM_THREADS;
    
    //初始版本
    // matMultCUDA<<<blocks * n, NUM_THREADS>>>
    //     (ac, n, bc, n, cc, n, n);
    
    //改良1.0
    // matMultCUDA<<<n, NUM_THREADS, sizeof(float) * n>>>
    //        (ac, n, bc, n, cc, n, n);

    //改良2.0
    // matMultCUDA<<<n, NUM_THREADS, sizeof(float) * n>>>
    //     (ac, pitch_a / sizeof(float), bc, pitch_b / sizeof(float),
    //     cc, pitch_c / sizeof(float), n);

    int bx = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blocks(bx, bx);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    matMultCUDA<<<blocks, threads>>>(ac, pitch_a / sizeof(float),
        bc, pitch_b / sizeof(float), cc, pitch_c / sizeof(float), n);
    
    // cudaMemcpy2D(c, sizeof(float) * ldc, cc, sizeof(float) * n,
    // sizeof(float) * n, n, cudaMemcpyDeviceToHost);

    cudaMemcpy2D(c, sizeof(float) * ldc, cc, pitch_c,
        sizeof(float) * n, n, cudaMemcpyDeviceToHost);


    cudaFree(ac);
    cudaFree(bc);
    cudaFree(cc);

    end = clock();

    return end - start;
}

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
