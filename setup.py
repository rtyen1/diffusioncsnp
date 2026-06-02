from setuptools import find_packages, setup

setup(
    name="ml2_meta_causal_discovery",
    version="0.0.1",
    packages=find_packages(
        include=[
            "ml2_meta_causal_discovery",
            "ml2_meta_causal_discovery.*",
        ]
    ),
)
