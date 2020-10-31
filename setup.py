from setuptools import setup, find_namespace_packages

setup(
    name='contactnets',
    packages=find_namespace_packages(include=['contactnets.*']),
    version='0.1',
    install_requires=[
            'pillow',
            'osqp',
            'mplcursors',
            'torch',
            'tensorflow==1.14.0',
            'tensorboard==2.1.0',
            'tensorboardX==1.9',
            'vpython',
            'torchtestcase',
            'pygame',
            'pyomo',
            'click',
            'sk-video',
            'pyopengl',
            'pyopengl_accelerate',
            'glfw',
            'gitpython',
            'psutil',
            'moviepy',
            'imageio',
            'tabulate',
            'grpcio'
            ])
