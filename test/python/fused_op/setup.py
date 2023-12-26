import setuptools


setuptools.setup(
    name="fused_op",
    version="0.2.4",
    author="Qingtao Li",
    author_email="qingtaoli@microsoft.com",
    description="A test package of fused operations.",
    long_description="A test package of fused operations. Including fused RMSNorm, RotaryEmbedding and LlamaMLP.",
    long_description_content_type="text/markdown",
    url="no url",
    packages=setuptools.find_packages(),
    package_data={"fused_op": ["kernel/*/*/*.json"]},
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Environment :: GPU :: NVIDIA CUDA :: 11.4"
    ],
    install_requires=[
        "antares==0.3.25.0"
    ]
)
