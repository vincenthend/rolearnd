# rolearnd
rolearnd is a mini Python module for machine learning

## Installation
### Dependencies
- Python 3
- Pip
- Libraries in `requirements.txt`

### User Installation
Install by installing the library required by using
```sh
pip install -r requirements.txt
```

## Development

### Setting up development environment
Virtualenv is required to keep requirements.txt clean from other unused requirements. Please read virtualenv guide for further reference

### Creating new features
Whenever creating new feature, you may create a new branch with the name format `feature/your_feature_name`
```sh
git checkout -b "feature/your_feature_name"
```

On merging with the main branch, rebase onto the main branch by using
```sh
git fetch
git rebase origin/master
```
and then creating a pull request on the repository webpage


### Adding a new library
After installation of library with pip, please renew `requirements.txt` by using
```sh
pip freeze > requirements.txt
```

### Naming convention
Refer to https://www.python.org/dev/peps/pep-0008 for naming convention

### Developing new classes
- Extend the previously made classes (e.g. `Classifier`) to add new classes
- If applicable for other, add utils to the `utils` submodule

### Final Note
- Prevent class from getting too big by using `utils`
- Don't forget to add **documentation** on how a class and algorithm works (or perhaps some usage example)