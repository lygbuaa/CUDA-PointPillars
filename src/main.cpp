/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime.h>

#include <string.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <dirent.h>

#include "pointpillar.hpp"
#include "common/check.hpp"

#define LOGPF(format, ...) fprintf(stderr ,"[%s:%d] " format "\n", __FILE__, __LINE__, ##__VA_ARGS__)

typedef struct
{
    float x;
    float y;
    float z;
    float intensity;
}PointXYZI_t;

typedef union 
{
    PointXYZI_t p;
    uint8_t buf[sizeof(PointXYZI_t)];
}PointXYZI_u;

void GetDeviceInfo(void)
{
  cudaDeviceProp prop;

  int count = 0;
  cudaGetDeviceCount(&count);
  printf("\nGPU has cuda devices: %d\n", count);
  for (int i = 0; i < count; ++i) {
    cudaGetDeviceProperties(&prop, i);
    printf("----device id: %d info----\n", i);
    printf("  GPU : %s \n", prop.name);
    printf("  Capbility: %d.%d\n", prop.major, prop.minor);
    printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
    printf("  Const memory: %luKB\n", prop.totalConstMem  >> 10);
    printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
    printf("  warp size: %d\n", prop.warpSize);
    printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
    printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  }
  printf("\n");
}

bool hasEnding(std::string const &fullString, std::string const &ending)
{
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

int getFolderFile(const char *path, std::vector<std::string>& files, const char *suffix = ".bin")
{
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(path)) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            std::string file = ent->d_name;
            if(hasEnding(file, suffix)){
                files.push_back(file.substr(0, file.length()-4));
            }
        }
        closedir(dir);
    } else {
        printf("No such folder: %s.", path);
        exit(EXIT_FAILURE);
    }
    return EXIT_SUCCESS;
}

int loadData(const char *file, void **data, unsigned int *length)
{
    std::fstream dataFile(file, std::ifstream::in);

    if (!dataFile.is_open()) {
        std::cout << "Can't open files: "<< file<<std::endl;
        return -1;
    }

    unsigned int len = 0;
    dataFile.seekg (0, dataFile.end);
    len = dataFile.tellg();
    dataFile.seekg (0, dataFile.beg);

    char *buffer = new char[len];
    if (buffer==NULL) {
        std::cout << "Can't malloc buffer."<<std::endl;
        dataFile.close();
        exit(EXIT_FAILURE);
    }

    dataFile.read(buffer, len);
    dataFile.close();

/** check the input pcd files */
#if 1
    const uint32_t points_count = len / sizeof(PointXYZI_t);
    const uint8_t* ptr_head = (uint8_t*)buffer;
    float x_max = -1.0f*1e8;
    float x_min = 1e8;
    float y_max = -1.0f*1e8;
    float y_min = 1e8;
    float z_max = -1.0f*1e8;
    float z_min = 1e8;
    float i_max = -1.0f*1e8;
    float i_min = 1e8;
    for(uint32_t idx=0; idx<points_count; idx++)
    {
        PointXYZI_u pu;
        const uint8_t* ptr = ptr_head + idx*sizeof(PointXYZI_t);
        memcpy(pu.buf, ptr, sizeof(PointXYZI_t));
        if(pu.p.x > x_max)
        {
            x_max = pu.p.x;
        }
        else if(pu.p.x < x_min)
        {
            x_min = pu.p.x;
        }

        if(pu.p.y > y_max)
        {
            y_max = pu.p.y;
        }
        else if(pu.p.y < y_min)
        {
            y_min = pu.p.y;
        }

        if(pu.p.z > z_max)
        {
            z_max = pu.p.z;
        }
        else if(pu.p.z < z_min)
        {
            z_min = pu.p.z;
        }

        if(pu.p.intensity > i_max)
        {
            i_max = pu.p.intensity;
        }
        else if(pu.p.intensity < i_min)
        {
            i_min = pu.p.intensity;
        }
    }

    LOGPF("pcd file %s points_count: %d, x-range: (%.3f, %.3f), y-range: (%.3f, %.3f), z-range: (%.3f, %.3f), i-range: (%.3f, %.3f)", \
        file, points_count, x_min, x_max, y_min, y_max, z_min, z_max, i_min, i_max);
#endif
    *data = (void*)buffer;
    *length = len;
    return 0;  
}

void SaveBoxPred(std::vector<pointpillar::lidar::BoundingBox> boxes, std::string file_name)
{
    std::ofstream ofs;
    ofs.open(file_name, std::ios::out);
    if (ofs.is_open()) {
        for (const auto box : boxes) {
          ofs << box.x << " ";
          ofs << box.y << " ";
          ofs << box.z << " ";
          ofs << box.w << " ";
          ofs << box.l << " ";
          ofs << box.h << " ";
          ofs << box.rt << " ";
          ofs << box.id << " ";
          ofs << box.score << "\n";
        }
    }
    else {
      std::cerr << "Output file cannot be opened!" << std::endl;
    }
    ofs.close();
    std::cout << "Saved prediction in: " << file_name << std::endl;
    return;
};

std::shared_ptr<pointpillar::lidar::Core> create_core() {
    pointpillar::lidar::VoxelizationParameter vp;
    vp.min_range = nvtype::Float3(0.0, -39.68f, -3.0);
    vp.max_range = nvtype::Float3(69.12f, 39.68f, 1.0);
    vp.voxel_size = nvtype::Float3(0.16f, 0.16f, 4.0f);
    vp.grid_size =
        vp.compute_grid_size(vp.max_range, vp.min_range, vp.voxel_size);
    vp.max_voxels = 40000;
    vp.max_points_per_voxel = 32;
    vp.max_points = 300000;
    vp.num_feature = 4;

    pointpillar::lidar::PostProcessParameter pp;
    pp.min_range = vp.min_range;
    pp.max_range = vp.max_range;
    pp.feature_size = nvtype::Int2(vp.grid_size.x/2, vp.grid_size.y/2);

    pointpillar::lidar::CoreParameter param;
    param.voxelization = vp;
    param.lidar_model = "../model/pointpillar.plan";
    param.lidar_post = pp;
    return pointpillar::lidar::create_core(param);
}

static bool startswith(const char *s, const char *with, const char **last)
{
    while (*s++ == *with++)
    {
        if (*s == 0 || *with == 0)
            break;
    }
    if (*with == 0)
        *last = s + 1;
    return *with == 0;
}

static void help()
{
    printf(
        "Usage: \n"
        "    ./pointpillar in/ out/ --timer\n"
        "    Run pointpillar inference with .bin under in, save .text under out\n"
        "    Optional: --timer, enable timer log\n"
    );
    exit(EXIT_SUCCESS);
}

int main(int argc, char** argv) {

    if (argc < 3 || argc > 4)
        help();

    const char *in_dir  = argv[1];
    const char *out_dir  = argv[2];

    const char *value = nullptr;
    bool timer = false;

    if (argc == 4) {
        if (startswith(argv[3], "--timer", &value)) {
            timer = true;
        }
    }

    GetDeviceInfo();

    std::vector<std::string> files;
    getFolderFile(in_dir, files);
    std::cout << "Total " << files.size() << std::endl;

    auto core = create_core();
    if (core == nullptr) {
        printf("Core has been failed.\n");
        return -1;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
  
    core->print();
    core->set_timer(timer);

    for (const auto & file : files)
    {
        std::string dataFile = std::string(in_dir) + file + ".bin";

        std::cout << "\n<<<<<<<<<<<" <<std::endl;
        std::cout << "Load file: "<< dataFile <<std::endl;

        //load points cloud
        unsigned int length = 0;
        void *data = NULL;
        std::shared_ptr<char> buffer((char *)data, std::default_delete<char[]>());
        loadData(dataFile.data(), &data, &length);
        buffer.reset((char *)data);
        int points_size = length/sizeof(float)/4;
        std::cout << "Lidar points count: "<< points_size <<std::endl;
    
        auto bboxes = core->forward((float *)buffer.get(), points_size, stream);
        std::cout<<"Detections after NMS: "<< bboxes.size()<<std::endl;

        std::string save_file_name = std::string(out_dir) + file + ".txt";
        SaveBoxPred(bboxes, save_file_name);

        std::cout << ">>>>>>>>>>>" << std::endl;
    }

    checkRuntime(cudaStreamDestroy(stream));
    return 0;
}
