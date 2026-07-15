from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sysconfig

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

import numpy as np


class NvccBuildExt(build_ext):
    def build_extension(self, ext: Extension) -> None:
        if ext.name != "cuda_resize_py":
            super().build_extension(ext)
            return

        import numpy as np

        project_root = pathlib.Path(__file__).resolve().parent

        output_path = pathlib.Path(self.get_ext_fullpath(ext.name)).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 确定conda，cuda，nvcc环境路径
        conda_prefix = pathlib.Path(
            os.environ.get("CONDA_PREFIX", sysconfig.get_config_var("prefix"))
        )
        cuda_home = pathlib.Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
        nvcc = shutil.which("nvcc") or str(cuda_home / "bin" / "nvcc")

        include_dirs = [
            project_root / "cuda",
            pathlib.Path(sysconfig.get_paths()["include"]),
            pathlib.Path(np.get_include()),
            conda_prefix / "include" / "opencv5",
            cuda_home / "include",
        ]

        library_dirs = [
            conda_prefix / "lib",
            cuda_home / "lib64",
        ]

        command = [
            nvcc,
            "-std=c++17",
            "-O3",
            "-allow-unsupported-compiler",
            "--compiler-options",
            "-fPIC",
            "-shared",
            str(project_root / "cuda" / "python_cuda_resize.cu"),
            "-o",
            str(output_path),
        ]

        for include_dir in include_dirs:
            command.extend(["-I", str(include_dir)])

        for library_dir in library_dirs:
            command.extend(["-L", str(library_dir)])

        command.extend([
            "-lopencv_core",
            "-lopencv_imgproc",
            "-lopencv_imgcodecs",
            "-lcudart",
            "-Xlinker",
            f"-rpath,{conda_prefix / 'lib'}",
            "-Xlinker",
            f"-rpath,{cuda_home / 'lib64'}",
        ])

        print("building cuda_resize_py with:")
        print(" ".join(command))
        subprocess.check_call(command)


setup(
    name="hpsc-cuda-resize",
    version="0.1.0",
    ext_modules=[
        Extension("cuda_resize_py", sources=["cuda/python_cuda_resize.cu"]),
        Extension(
            "fast_cpu_resize",
            sources=["cpu/python_fast_cpu_resize.cpp"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-std=c++17", "-O3", "-march=native", "-pthread"],
            extra_link_args=["-pthread"],
        ),
    ],
    cmdclass={"build_ext": NvccBuildExt},
)
