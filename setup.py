from setuptools import setup, find_packages

setup(
    name="rayleigh-quotient",
    description="multidimensional rayleigh quotient",
    version="0.1.0",
    author="chep0k",
    author_email="chep0k@mail.ru",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "krypy",
    ],
)
