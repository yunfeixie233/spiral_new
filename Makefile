print-%: ; @echo $* = $($*)
SHELL=/bin/bash
PROJECT_NAME  = spiral
COPYRIGHT     = "SPIRAL Team. All Rights Reserved."
PROJECT_PATH  = spiral
SOURCE_FOLDERS = $(PROJECT_PATH) train_spiral.py run.sh
LINT_PATHS    = ${PROJECT_PATH} evals
LINT_PATHS_NO_EVAL = ${PROJECT_PATH} train_spiral.py

check_install = python3 -c "import $(1)" || pip3 install $(1) --upgrade
check_install_extra = python3 -c "import $(1)" || pip3 install $(2) --upgrade

go-install:
	# requires go >= 1.16
	command -v go || (sudo apt-get install -y golang && sudo ln -sf /usr/lib/go/bin/go /usr/bin/go)

addlicense-install: go-install
	command -v addlicense || go install github.com/google/addlicense@latest

test:
	$(call check_install, pytest)
	pytest -s

lint:
	$(call check_install, isort)
	$(call check_install, pylint)
	isort --check --diff --project=${LINT_PATHS}
	pylint -j 8 --recursive=y ${LINT_PATHS}

format_all:
	$(call check_install, autoflake)
	autoflake --remove-all-unused-imports -i -r ${LINT_PATHS}
	$(call check_install, black)
	black ${LINT_PATHS}
	$(call check_install, isort)
	isort ${LINT_PATHS}

format:
	$(call check_install, autoflake)
	autoflake --remove-all-unused-imports -i -r ${LINT_PATHS_NO_EVAL}
	$(call check_install, black)
	black ${LINT_PATHS_NO_EVAL}
	$(call check_install, isort)
	isort ${LINT_PATHS_NO_EVAL}

check-docstyle:
	$(call check_install, pydocstyle)
	pydocstyle ${PROJECT_PATH} --convention=google

checks: lint check-docstyle

clean:
	@-rm -rf build/ dist/ .eggs/ site/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	@-find . -name '*.pyc' -type f -exec rm -rf {} +
	@-find . -name '__pycache__' -exec rm -rf {} +

package: clean
	PRODUCTION_MODE=yes python setup.py bdist_wheel

publish: package
	twine upload dist/*

addlicense: addlicense-install
	$$(command -v addlicense || echo $(HOME)/go/bin/addlicense) -c $(COPYRIGHT) -ignore tests/coverage.xml -l apache -y 2025 $(SOURCE_FOLDERS)

check-license: addlicense-install
	$$(command -v addlicense || echo $(HOME)/go/bin/addlicense) -c $(COPYRIGHT) -ignore tests/coverage.xml -l apache -y 2025 -check $(SOURCE_FOLDERS)

.PHONY: format lint check-docstyle checks
