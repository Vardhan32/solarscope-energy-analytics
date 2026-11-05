from setuptools import setup, find_packages

setup(
    name="solarscope",
    version="1.0.0",
    description="Solar Power Generation Analytics & Forecasting Platform",
    author="Vardhan32",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
        "streamlit>=1.35.0",
        "plotly>=5.20.0",
        "pyarrow>=15.0.0",
        "joblib>=1.3.0",
        "pytest>=8.0.0",
    ],
)
