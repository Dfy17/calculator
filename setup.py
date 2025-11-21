from setuptools import setup, find_packages

setup(
    name="gastrointestinal_tumor_calculator",
    version="0.1",
    packages=find_packages(),
    python_requires=">=3.9, <3.13",  # 限制 Python 版本（避开 3.13，用稳定的 3.9-3.12）
    install_requires=[
        "numpy>=1.22.0, <1.26.0",  # 放宽版本范围，提高兼容性
        "pandas>=1.4.0, <1.6.0",
        "scipy>=1.8.0, <1.11.0",
        "tableone>=0.12.0, <0.14.0",
        "matplotlib>=3.5.0, <3.8.0",
        "seaborn>=0.11.0, <0.13.0",
        "xgboost>=1.5.0, <1.8.0",
        "lightgbm>=3.2.0, <3.4.0",
        "scikit-learn>=1.0.0, <1.3.0",
        "lime>=0.2.0, <0.2.1",
        "streamlit>=1.28.0, <1.35.0",
    ],
)