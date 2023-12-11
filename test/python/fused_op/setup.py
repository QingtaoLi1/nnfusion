import os
import setuptools
from setuptools.command.install import install
import subprocess

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        print("Running post-install script...")

        os.system('BACKEND=c-cuda antares torch-setup')
        os.system('BACKEND=c-mcpu antares torch-setup')

        # env = os.environ.copy()
        # env["BACKEND"] = "c-cuda"
        # try:
        #     subprocess.run(['antares', 'torch-setup'], env=env, check=True)
        # except subprocess.CalledProcessError as e:
        #     print("An error occurred while setting up environment variables. The failed command is:")
        #     print('>>> BACKEND=c-mcpu antares torch-setup')
        #     print(str(e))

        # env["BACKEND"] = "c-mcpu"
        # try:
        #     subprocess.run(['antares', 'torch-setup'], env=env, check=True)
        # except subprocess.CalledProcessError as e:
        #     print("An error occurred while setting up environment variables. The failed command is:")
        #     print('>>> BACKEND=c-mcpu antares torch-setup')
        #     print(str(e))

setuptools.setup(
    name="fused_op",
    version="0.1",
    author="Qingtao Li",
    author_email="qingtaoli@microsoft.com",
    description="A test package of fused operations.",
    long_description="A test package of fused operations.",
    long_description_content_type="text/markdown",
    url="no url",
    packages=setuptools.find_packages(),
    package_data={"fused_op": ["kernel/*/*.json"]},
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Environment :: GPU :: NVIDIA CUDA :: 11.4"
    ],
    install_requires=[
        "antares==0.3.9.0"
    ],
    cmdclass={
        'install': PostInstallCommand,
    }
)
