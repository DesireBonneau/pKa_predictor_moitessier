from setuptools import setup, find_packages

setup(
    name='pka_predictor_moitessier',
    version='0.1.9',
    author="Moitessier Lab",
    author_email="nicolas.moitessier@mcgill.ca",
    description="Graph-based pKa prediction for small molecules",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MoitessierLab/pKa-predictor",   # TO EDIT TO MY OWN GITHUB REPO
    license="GPL-3.0",
    keywords="pKa prediction GNN chemistry rdkit",
    packages=find_packages(include=['pka_predictor', 'pka_predictor.*']),
    package_data={
        "pka_predictor": ["Model/*.pth"],  # if under pka_predictor/Model/
    },
    install_requires=[
        'torch',
        'torch_geometric',
        'pandas',
        'numpy',
        'rdkit',
        'seaborn',
        'hyperopt',
        'scikit-learn',
    ],
    entry_points={
        "console_scripts": [
            "pka-predictor = pka_predictor.__main__:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    python_requires='>=3.8',
)
