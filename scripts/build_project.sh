# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC 
#  * @Author: lijinwen 
#  * @Date: 2022-10-26 09:42:14 
#  * @Last Modified by: lijinwen 
#  * @Last Modified time: 2022-10-26 09:42:14 
#  **************************************************************/

error_log(){
  echo -e "\033[31mERROR:$1\033[0m"
  exit 1
}

if [ -z "$#" ];then
    echo -e "\nUsage ./scripts/build_project.sh x86_64 \n"
    exit 1
fi

export CURRENT_PATH=${PWD}
export PATH=$PATH:/usr/local/cuda/bin

dirname=./build

if [ ! -d ${dirname} ]; then
  mkdir -m 777 ${dirname}
fi

rm -rf ${dirname}/*

if [ -d Release ]; then
  rm -rf Release
fi

if [ -d targetfs ]; then
  rm -rf targetfs
fi

cd ${dirname}

if [ $# -eq 1 ] && [ "$1" = "x86_64" ]; then
  echo "-- This is x86_64 building project!"

  cmake ..  -DUSE_DOCKER=ON  -DCMAKE_BUILD_TYPE=Release
  make -j$(nproc) && make -j$(nproc) install 

elif [ $# -eq 1 ] && [ "$1" = "aarch64" ]; then
  echo "-- This is aarch64 building project!"

  error_log "not support aarch64, exit!"
  exit 1
else
  error_log "wrong command, exit!"
  exit 1
fi

cp -fr ../third_party/onnxruntime-linux-x64-1.8.1/lib/*   ../${dirname}/

