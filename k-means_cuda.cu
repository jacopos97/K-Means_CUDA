#include <iostream>
#include "INIReader.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <unordered_map>
#include <cuda_runtime_api.h>

using namespace std;
using namespace chrono;

static const int POINTS_NUMBER = 40000000;
static const int CLUSTER_NUMBER = 4;
static const string CONFIG_FILE_PATH = "../config_files/config_sets.ini";
static const string DESIRED_CONFIG = to_string(CLUSTER_NUMBER) + "_cluster";
static const int ITERATION_NUMBER = 10;
static const int THREADS_PER_BLOCK = 128;

struct DataPoints {
    float* x;
    float* y;
    float* z;
};

void printCentroids(DataPoints& centroids);
string getNumberString(int number);
bool readDatasetFromFile(DataPoints& dataset);
bool initializeCentroids(DataPoints& centroids, const string& configFilePath, const string& desiredConfig);
void freeDataPoints(DataPoints &dataPoints);
void freeDevDataPoints(float *&devPointX, float *&devPointY, float *&devPointZ);
void allocPointDevMemory(float *&hostPointX, float *&hostPointY, float *&hostPointZ, float *&devPointX, float *&devPointY, float *&devPointZ, int size);
__global__ void assignPointToCluster(const float* devPointX, const float* devPointY, const float* devPointZ, float* devCentroidX, float* devCentroidY, float* devCentroidZ, float* devNewCentroidX, float* devNewCentroidY, float* devNewCentroidZ, float* devClustersSize);
//__global__ void assignPointToCluster(float *&devPointX, float *&devPointY, float *&devPointZ, float *&devCentroidX, float *&devCentroidY, float *&devCentroidZ, float *&devNewCentroidX, float *&devNewCentroidY, float *&devNewCentroidZ, float *&devClustersSize);

int main() {

    DataPoints dataPoints{};
    if(!readDatasetFromFile(dataPoints)) return -1;
    /*for(int i=0; i<POINTS_NUMBER; i++){
        cout << "(" << dataPoints.x[i] << ", " << dataPoints.y[i] << ", " << dataPoints.z[i] << ")" << endl;
    }*/

    float *devPointX, *devPointY, *devPointZ;
    allocPointDevMemory(dataPoints.x, dataPoints.y, dataPoints.z, devPointX, devPointY, devPointZ, POINTS_NUMBER);
    /*auto* x = new float[POINTS_NUMBER];
    auto* y = new float[POINTS_NUMBER];
    auto* z = new float[POINTS_NUMBER];
    cudaMemcpy(x, devPointX, POINTS_NUMBER*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(y, devPointY, POINTS_NUMBER*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(z, devPointZ, POINTS_NUMBER*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0; i<POINTS_NUMBER; i++) {
        cout<<x[i]<<" "<<y[i]<<" "<<z[i]<<" "<<endl;
    }
    delete[] x,y,z;*/

    DataPoints centroids{};
    if (!initializeCentroids(centroids, CONFIG_FILE_PATH, DESIRED_CONFIG)) return -1;
    printCentroids(centroids);

    float *devCentroidX, *devCentroidY, *devCentroidZ;
    allocPointDevMemory(centroids.x, centroids.y, centroids.z, devCentroidX, devCentroidY, devCentroidZ, CLUSTER_NUMBER);
    /*auto* x = new float[CLUSTER_NUMBER];
    auto* y = new float[CLUSTER_NUMBER];
    auto* z = new float[CLUSTER_NUMBER];
    cudaMemcpy(x, devCentroidX, CLUSTER_NUMBER*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(y, devCentroidY, CLUSTER_NUMBER*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(z, devCentroidZ, CLUSTER_NUMBER*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0; i<CLUSTER_NUMBER; i++) {
        cout<<x[i]<<" "<<y[i]<<" "<<z[i]<<" "<<endl;
    }
    delete[] x,y,z;*/

    auto* defaultArray = new float[CLUSTER_NUMBER]();
    auto* clustersSize = new float[CLUSTER_NUMBER]();
    float* devClustersSize;
    cudaMalloc((void**)&devClustersSize, CLUSTER_NUMBER*sizeof(float));
    cudaMemcpy(devClustersSize, defaultArray, CLUSTER_NUMBER*sizeof(float), cudaMemcpyHostToDevice);
    /*auto* x = new float[CLUSTER_NUMBER];
    cudaMemcpy(x, devClustersSize, CLUSTER_NUMBER*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0; i<CLUSTER_NUMBER; i++) {
        cout<<x[i]<<endl;
    }
    delete[] x;*/

    float *devNewCentroidX, *devNewCentroidY, *devNewCentroidZ;
    allocPointDevMemory(defaultArray, defaultArray, defaultArray, devNewCentroidX, devNewCentroidY, devNewCentroidZ, CLUSTER_NUMBER);
    /*auto* x = new float[CLUSTER_NUMBER];
    auto* y = new float[CLUSTER_NUMBER];
    auto* z = new float[CLUSTER_NUMBER];
    cudaMemcpy(x, devNewCentroidX, CLUSTER_NUMBER*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(y, devNewCentroidY, CLUSTER_NUMBER*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(z, devNewCentroidZ, CLUSTER_NUMBER*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0; i<CLUSTER_NUMBER; i++) {
        cout<<x[i]<<" "<<y[i]<<" "<<z[i]<<" "<<endl;
    }
    delete[] x;
    delete[] y;
    delete[] z;*/

    /*DataPoints newCentroids{};
    newCentroids.x = (float*)malloc(CLUSTER_NUMBER * sizeof(float));
    newCentroids.y = (float*)malloc(CLUSTER_NUMBER * sizeof(float));
    newCentroids.z = (float*)malloc(CLUSTER_NUMBER * sizeof(float));*/


    auto startTime = high_resolution_clock::now();

    for (int iteration = 0; iteration < ITERATION_NUMBER; iteration++) {
        cout << endl << "Iteration " << iteration + 1 << ":" << endl;

        dim3 dimBlock(THREADS_PER_BLOCK);
        dim3 dimGrid((POINTS_NUMBER+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK);
        assignPointToCluster <<<dimGrid,dimBlock,7*CLUSTER_NUMBER*sizeof(float)>>> (devPointX,devPointY,devPointZ,devCentroidX,devCentroidY,devCentroidZ,devNewCentroidX,devNewCentroidY,devNewCentroidZ,devClustersSize);
        //assignPointToCluster <<<dimGrid,dimBlock>>> (devPointX,devPointY,devPointZ,devCentroidX,devCentroidY,devCentroidZ,devClustersSize);
        cudaDeviceSynchronize();
        /*auto* a = new float[CLUSTER_NUMBER];
        auto* b = new float[CLUSTER_NUMBER];
        auto* c = new float[CLUSTER_NUMBER];
        cudaMemcpy(a, devCentroidX, CLUSTER_NUMBER*sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(b, devCentroidY, CLUSTER_NUMBER*sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(c, devCentroidZ, CLUSTER_NUMBER*sizeof(float),cudaMemcpyDeviceToHost);
        for(int i=0; i<CLUSTER_NUMBER; i++) {
            cout<<a[i]<<" "<<b[i]<<" "<<c[i]<<" "<<endl;
        }
        delete[] a;
        delete[] b;
        delete[] c;*/

        cudaMemcpy(centroids.x, devNewCentroidX, CLUSTER_NUMBER * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(centroids.y, devNewCentroidY, CLUSTER_NUMBER * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(centroids.z, devNewCentroidZ, CLUSTER_NUMBER * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(clustersSize,devClustersSize, CLUSTER_NUMBER*sizeof(float),cudaMemcpyDeviceToHost);
        /*printCentroids(newCentroids);
        cout<<endl;
        for (int i = 0; i < CLUSTER_NUMBER; i++) {
            cout<<clustersSize[i]<<" ";
        }*/

        for (int i = 0; i < CLUSTER_NUMBER; i++) {
            centroids.x[i] = centroids.x[i] / clustersSize[i];
            centroids.y[i] = centroids.y[i] / clustersSize[i];
            centroids.z[i] = centroids.z[i] / clustersSize[i];
        }

        cudaMemcpy(devCentroidX, centroids.x, CLUSTER_NUMBER * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(devCentroidY, centroids.y, CLUSTER_NUMBER * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(devCentroidZ, centroids.z, CLUSTER_NUMBER * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(devNewCentroidX, defaultArray, CLUSTER_NUMBER*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(devNewCentroidY, defaultArray, CLUSTER_NUMBER*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(devNewCentroidZ, defaultArray, CLUSTER_NUMBER*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(devClustersSize, defaultArray, CLUSTER_NUMBER*sizeof(float), cudaMemcpyHostToDevice);

        cout << endl;
        for (int i = 0; i < CLUSTER_NUMBER; i++) {
            cout << "Cluster" << i + 1 << " size: " << static_cast<int>(clustersSize[i]) << endl;
        }

        cout << endl;
        printCentroids(centroids);
    }

    auto endTime = high_resolution_clock::now();
    auto time = duration_cast<microseconds>(endTime - startTime).count() / 1000.f;
    cout << "Duration: " << time << " ms" << endl;

    freeDataPoints(dataPoints);
    freeDevDataPoints(devPointX,devPointY,devPointZ);
    freeDataPoints(centroids);
    freeDevDataPoints(devCentroidX,devCentroidY,devCentroidZ);
    delete[] defaultArray;
    delete[] clustersSize;
    freeDevDataPoints(devNewCentroidX,devNewCentroidY,devNewCentroidZ);
    cudaFree(devClustersSize);
    devClustersSize = nullptr;
    //freeDataPoints(newCentroids);

    return 0;
}

void allocPointDevMemory(float *&hostPointX, float *&hostPointY, float *&hostPointZ, float *&devPointX, float *&devPointY, float *&devPointZ, int size) {
    cudaMalloc((void**)&devPointX, size * sizeof(float));
    cudaMalloc((void**)&devPointY, size * sizeof(float));
    cudaMalloc((void**)&devPointZ, size * sizeof(float));
    cudaMemcpy(devPointX, hostPointX, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devPointY, hostPointY, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devPointZ, hostPointZ, size * sizeof(float), cudaMemcpyHostToDevice);
}

void freeDataPoints(DataPoints &dataPoints) {
    free(dataPoints.x);
    free(dataPoints.y);
    free(dataPoints.z);
    dataPoints.x = nullptr;
    dataPoints.y = nullptr;
    dataPoints.z = nullptr;
}

void freeDevDataPoints(float *&devPointX, float *&devPointY, float *&devPointZ){
    cudaFree(devPointX);
    cudaFree(devPointY);
    cudaFree(devPointZ);
    devPointX = nullptr;
    devPointY = nullptr;
    devPointZ = nullptr;
}

bool readDatasetFromFile(DataPoints& dataset) {
    dataset.x = (float*)malloc(POINTS_NUMBER* sizeof(float));
    dataset.y = (float*)malloc(POINTS_NUMBER* sizeof(float));
    dataset.z = (float*)malloc(POINTS_NUMBER* sizeof(float));
    string path = "../datasets/generated_blob_dataset_" + getNumberString(POINTS_NUMBER) + ".csv";
    ifstream file(path);
    if (file.is_open()) {
        string line;
        cout << "Reading the dataset..." << endl;
        int i = 0;
        while (getline(file, line)) {
            istringstream coordinates(line);
            float x;
            float y;
            float z;
            char delimiter1;
            char delimiter2;
            if (coordinates >> x >> delimiter1 >> y >> delimiter2 >> z) {
                dataset.x[i] = x;
                dataset.y[i] = y;
                dataset.z[i] = z;
                i++;
            }
        }
        file.close();
        cout << "Dataset loaded from " << path << endl;
        return true;
    } else {
        cerr << "Error: Unable to open file " << path << endl;
        freeDataPoints(dataset);
        return false;
    }
}

string getNumberString(int number) {
    switch (number) {
        case 4000:
            return "4k";
        case 40000:
            return "40k";
        case 400000:
            return "400k";
        case 4000000:
            return "4m";
        case 40000000:
            return "40m";
        default:
            return "";
    }
}

bool initializeCentroids(DataPoints& centroids, const string& configFilePath, const string& desiredConfig) {
    INIReader reader(configFilePath);
    if (reader.ParseError() < 0) {
        cerr << "Error loading config file\n";
        return false;
    }
    centroids.x = (float*)malloc(CLUSTER_NUMBER * sizeof(float));
    centroids.y = (float*)malloc(CLUSTER_NUMBER * sizeof(float));
    centroids.z = (float*)malloc(CLUSTER_NUMBER * sizeof(float));
    for(int i=0; i < CLUSTER_NUMBER; i++)  {
        istringstream coordinates(reader.Get(desiredConfig, "centroid" + to_string(i), ""));
        float x;
        float y;
        float z;
        char delimiter1;
        char delimiter2;
        if (coordinates >> x >> delimiter1 >> y >> delimiter2 >> z){
            centroids.x[i] = x;
            centroids.y[i] = y;
            centroids.z[i] = z;
        }
    }
    return true;
}

void printCentroids(DataPoints& centroids) {
    for (int i=0; i<CLUSTER_NUMBER; i++) {
        cout << "(" << centroids.x[i] << ", " << centroids.y[i] << ", " << centroids.z[i] << ")" << endl;
    }
}

__global__ void assignPointToCluster(const float* devPointX, const float* devPointY, const float* devPointZ, float* devCentroidX, float* devCentroidY, float* devCentroidZ, float* devNewCentroidX, float* devNewCentroidY, float* devNewCentroidZ, float* devClustersSize) {

    extern __shared__ float sharedMem[];

    int tid = blockIdx.x*blockDim.x+threadIdx.x;

    /*if (tid < POINTS_NUMBER){
        printf("Thread %d: Hello from thread %d in block %d\n", tid, threadIdx.x, blockIdx.x);
        //devClustersSize[tid] = tid;
        //printf("%f\n", devClustersSize[tid]);
    }*/
    if (threadIdx.x == 0) {
        for (int i = 0; i < CLUSTER_NUMBER; i++) {
            sharedMem[i] = devCentroidX[i];
            sharedMem[CLUSTER_NUMBER + i] = devCentroidY[i];
            sharedMem[2*CLUSTER_NUMBER + i] = devCentroidZ[i];
            sharedMem[3*CLUSTER_NUMBER + i] = 0;
            sharedMem[4*CLUSTER_NUMBER + i] = 0;
            sharedMem[5*CLUSTER_NUMBER + i] = 0;
            sharedMem[6*CLUSTER_NUMBER + i] = 0;
        }
    }

    __syncthreads();

    if (tid < POINTS_NUMBER) {
        float x = devPointX[tid];
        float y = devPointY[tid];
        float z = devPointZ[tid];
        float shortest_distance = sqrt(
                pow(sharedMem[0] - x, 2) + pow(sharedMem[CLUSTER_NUMBER] - y, 2) +
                pow(sharedMem[2*CLUSTER_NUMBER] - z, 2));
        int clusterType = 0;
        for (int i = 1; i < CLUSTER_NUMBER; i++) {
            float centroid_distance = sqrt(
                    pow(sharedMem[i] - x, 2) + pow(sharedMem[CLUSTER_NUMBER + i] - y, 2) +
                    pow(sharedMem[2*CLUSTER_NUMBER + i] - z, 2));
            if (centroid_distance < shortest_distance) {
                shortest_distance = centroid_distance;
                clusterType = i;
            }
        }
        atomicAdd(&(sharedMem[3*CLUSTER_NUMBER + clusterType]), x);
        atomicAdd(&(sharedMem[4*CLUSTER_NUMBER + clusterType]), y);
        atomicAdd(&(sharedMem[5*CLUSTER_NUMBER + clusterType]), z);
        atomicAdd(&(sharedMem[6*CLUSTER_NUMBER + clusterType]), 1);
    }

    __syncthreads();

    if (threadIdx.x==0 && tid<POINTS_NUMBER) {
        for (int i = 0; i < CLUSTER_NUMBER; i++) {
            atomicAdd(&(devNewCentroidX[i]), sharedMem[3*CLUSTER_NUMBER + i]);
            atomicAdd(&(devNewCentroidY[i]), sharedMem[4*CLUSTER_NUMBER + i]);
            atomicAdd(&(devNewCentroidZ[i]), sharedMem[5*CLUSTER_NUMBER + i]);
            atomicAdd(&(devClustersSize[i]), sharedMem[6*CLUSTER_NUMBER + i]);
            sharedMem[3*CLUSTER_NUMBER + i] = 0;
            sharedMem[4*CLUSTER_NUMBER + i] = 0;
            sharedMem[5*CLUSTER_NUMBER + i] = 0;
            sharedMem[6*CLUSTER_NUMBER + i] = 0;
        }
    }
}