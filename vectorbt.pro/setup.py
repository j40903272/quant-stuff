from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8", errors="ignore") as fh:
    long_description = fh.read()

version = {}
with open("vectorbtpro/_version.py", encoding="utf-8") as fp:
    exec(fp.read(), version)


def get_extra_requires(path, add_all=True):
    """Parse extra requirements."""
    import re
    from collections import defaultdict

    with open(path, encoding="utf-8") as fp:
        extra_deps = defaultdict(set)
        for k in fp:
            if k.strip() and not k.startswith("#"):
                tags = set()
                if ":" in k:
                    k, v = k.split(":")
                    tags.update(vv.strip() for vv in v.split(","))
                tags.add(re.split("[<=>]", k)[0])
                for t in tags:
                    extra_deps[t].add(k)

        # add tag `all` at the end
        if add_all:
            extra_deps["full"] = set(vv for v in extra_deps.values() for vv in v)

    return extra_deps


setup(
    name="vectorbtpro",
    version=version["__version__"],
    description="Next-Generation Quantitative Analysis Tool",
    author="Oleg Polakow",
    author_email="olegpolakow@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/polakowo/vectorbt.pro",
    packages=find_packages(),
    package_data={"vectorbtpro": ["templates/*.json"]},
    python_requires="~=3.8",
    license="LICENSE.md",
    data_files=[("", ["LICENSE.md"])],
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.5.0",
        "numba>=0.53.1, <0.57.0; python_version<'3.10'",
        "numba>=0.56.0, <0.57.0; python_version>='3.10' and python_version<'3.11'",
        "numba>=0.57.0; python_version>='3.11'",
        "scipy",
        "scikit-learn",
        "schedule",
        "requests",
        "tqdm",
        "python-dateutil",
        "dateparser",
        "imageio",
        "mypy_extensions",
        "humanize",
        "attrs>=21.1.0",
        "websocket-client",
        "graphlib_backport; python_version<'3.9'",
    ],
    extras_require=get_extra_requires("extra-requirements.txt"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Software Development",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)
