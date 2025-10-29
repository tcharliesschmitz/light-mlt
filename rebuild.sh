rm -rf dist build src/*.egg-info
python3 -m build
twine check dist/*
twine upload dist/*
