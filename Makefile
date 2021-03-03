.PHONY: clean
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete

.PHONY: lint
lint:
	flake8 categorical_encoding && isort --check-only --recursive categorical_encoding

.PHONY: lint-fix
lint-fix:
	autopep8 --in-place --recursive --max-line-length=100 --select="E225,E303,E302,E203,E128,E231,E251,E271,E127,E126,E301,W291,W293,E226,E306,E221" categorical_encoding
	isort --recursive categorical_encoding

.PHONY: test
test:
	pytest categorical_encoding/tests ${addopts}

.PHONY: testcoverage
testcoverage: lint
	pytest categorical_encoding/tests --cov=categorical_encoding

.PHONY: installdeps
installdeps:
	pip install --upgrade pip
	pip install -e .
	pip install -r dev-requirements.txt


