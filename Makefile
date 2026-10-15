#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = TC5061-MiBici-Equipo50
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER ?= python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################


.PHONY: check-python
check-python:
	@$(PYTHON_INTERPRETER) -c 'import sys; expected=(3, 12); actual=sys.version_info[:2]; assert actual == expected, "Este proyecto requiere Python 3.12; se detectó %s.%s" % actual'

## Install Python dependencies
.PHONY: requirements
requirements: check-python
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 mibici_ml
	isort --check --diff mibici_ml
	black --check mibici_ml

## Format source code with black
.PHONY: format
format:
	isort mibici_ml
	black mibici_ml





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) -m mibici_ml.dataset download-drive
	$(PYTHON_INTERPRETER) -m mibici_ml.dataset profile

## Train and track the P2 baseline in MLflow
.PHONY: train-p2
train-p2:
	$(PYTHON_INTERPRETER) -m mibici_ml.modeling.train p2-baseline

## Open the local MLflow tracking interface
.PHONY: mlflow-ui
mlflow-ui: check-python
	$(PYTHON_INTERPRETER) -m mlflow ui --backend-store-uri sqlite:///mlflow.db


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
