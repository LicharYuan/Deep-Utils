import setuptools, os


with open("README.md", "r") as fh:
    long_description = fh.read()

def get_install_requires():
    install_requires = [
        'opencv-python',
        'termcolor',
        'pyyaml',
        'tabulate',
        'imageio',
        'matplotlib',
        # 'mxnet',
        'scikit-image'
    ]

    return install_requires

setuptools.setup(
    name="TUtils", 
    version="0.0.1",
    author="Liuchun Yuan",
    author_email="ylc0003@gmail.com",
    description="Provide utils for Deep",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires = get_install_requires(),
)

