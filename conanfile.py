# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC 
#  * @Author: lijinwen 
#  * @Date: 2022-10-26 09:42:14 
#  * @Last Modified by: lijinwen 
#  * @Last Modified time: 2022-10-26 09:42:14 
#  **************************************************************/

from conans import ConanFile, CMake, tools

codePath = "/root/code/xop"

class TensorrtinferConan(ConanFile):
    name = "xop"
    version = "1.0.0-alpha.1"
    license = "<Put the package license here>"
    author = "lijinwen@saicmotor.com"
    url = "http://10.94.119.155/algorithm_optimization/xop.git"
    topics = ("json", "conan")
    settings =  "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}
    generators = "cmake"

    exports_sources = "CMakeLists.txt"

    def build(self):
      self.run('./scripts/build_project.sh %s'%(self.settings.arch), cwd=codePath)

    def package(self):
      # xop
      self.copy("*.so*", dst="lib/%s"%(self.settings.arch), src="%s/Release/lib/"%(codePath),  keep_path=False, symlinks=True)
      
      self.copy("*.so*", dst="%s/targetfs/lib/%s"%(codePath, self.settings.arch),src="%s/Release/lib/"%(codePath),  keep_path=False, symlinks=True)
      
      # # xop test
      self.copy("*", dst="%s/targetfs/test/"%(codePath), src="%s/src/test"%(codePath))
      self.copy("*.so*", dst="%s/targetfs/test/lib/"%(codePath),src="%s/third_party/onnxruntime-linux-x64-1.8.1/lib"%(codePath),  keep_path=False, symlinks=True)

      # self.copy("*.*", dst="%s/targetfs/test/data"%(codePath), src="%s/data/"%(codePath))
      

    def package_info(self):
        self.cpp_info.libs = [self.name]