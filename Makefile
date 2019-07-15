.PHONY: lint
lint:
	flake8 featuretools && isort --check-only --recursive *.ipynb

.PHONY: lint-fix
lint-fix:
	autopep8 --in-place --recursive --max-line-length=100 --exclude="*/migrations/*" --select="E225,E303,E302,E203,E128,E231,E251,E271,E127,E126,E301,W291,W293,E226,E306,E221" *.ipynb
	isort --recursive *.ipynb
