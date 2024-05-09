
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mandelbrother",
    version="0.0.1",
    author="Ronald Paul Jenkins",
    author_email="ronald.paul.jenkins@gmail.com",
    description="GPU Mandelbrot set visualizer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires = [
        'numpy',
        'matplotlib',
        'scipy',
        'moderngl',
        'moderngl_window',
        ],
    url="https://github.com/poeface/mandelbrother",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    # package_data={'futhark-tracer':['assets/*']},
)
