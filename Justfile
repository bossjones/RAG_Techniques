set shell := ["zsh", "-cu"]

# just manual: https://github.com/casey/just/#readme

# Ignore the .env file that is only used by the web service
set dotenv-load := false

CURRENT_DIR := "$(pwd)"



base64_cmd := if "{{os()}}" == "macos" { "base64 -w 0 -i cert.pem -o ca.pem" } else { "base64 -w 0 -i cert.pem > ca.pem" }
grep_cmd := if "{{os()}}" =~ "macos" { "ggrep" } else { "grep" }

# List all available just commands
_default:
		@just --list

# Print the current operating system
info:
		print "OS: {{os()}}"

# Display system information
system-info:
	@echo "CPU architecture: {{ arch() }}"
	@echo "Operating system type: {{ os_family() }}"
	@echo "Operating system: {{ os() }}"

# verify python is running under pyenv
which-python:
		python -c "import sys;print(sys.executable)"

# when developing, you can use this to watch for changes and restart the server
autoreload-code:
	watchmedo auto-restart --pattern "*.py" --recursive --signal SIGTERM goobctl go

# Open the HTML coverage report in the default
local-open-coverage:
	./scripts/open-browser.py file://${PWD}/htmlcov/index.html

# Open the HTML coverage report in the default
open-coverage: local-open-coverage

# Run unit tests and open the coverage report
local-unittest:
	bash scripts/unittest-local
	./scripts/open-browser.py file://${PWD}/htmlcov/index.html

# Fetch multiple Python versions using rye
rye-get-pythons:
	rye fetch 3.8.19
	rye fetch 3.9.19
	rye fetch 3.10.14
	rye fetch 3.11.4
	rye fetch 3.12.3

# Add all dependencies using a custom script
rye-add-all:
	./contrib/rye-add-all.sh

# Run all pre-commit hooks on all files
pre-commit-run-all:
	pre-commit run --all-files

# Install pre-commit hooks
pre-commit-install:
	pre-commit install

# Display the dependency tree of the project
pipdep-tree:
	pipdeptree --python .venv/bin/python3

# install rye tools globally
rye-tool-install:
	rye install invoke
	rye install pipdeptree
	rye install click

# Lint GitHub Actions workflow files
lint-github-actions:
	actionlint

# check that taplo is installed to lint/format TOML
check-taplo-installed:
	@command -v taplo >/dev/null 2>&1 || { echo >&2 "taplo is required but it's not installed. run 'brew install taplo'"; exit 1; }

# Format Python files using pre-commit
fmt-python:
	git ls-files '*.py' '*.ipynb' | xargs pre-commit run --files

# Format Markdown files using pre-commit
fmt-markdown-pre-commit:
	git ls-files '*.md' | xargs pre-commit run --files

# format pyproject.toml using taplo
fmt-toml:
	pre-commit run taplo-format --all-files

# SOURCE: https://github.com/PovertyAction/ipa-data-tech-handbook/blob/ed81492f3917ee8c87f5d8a60a92599a324f2ded/Justfile

# Format all markdown and config files
fmt-markdown:
	git ls-files '*.md' | xargs mdformat

# Format a single markdown file, "f"
fmt-md f:
	mdformat {{ f }}

# format all code using pre-commit config
fmt: fmt-python fmt-toml

# lint python files using ruff
lint-python:
	pre-commit run ruff --all-files

# lint TOML files using taplo
lint-toml: check-taplo-installed
	pre-commit run taplo-lint --all-files

# lint yaml files using yamlfix
lint-yaml:
	pre-commit run yamlfix --all-files

# lint pyproject.toml and detect log_cli = true
lint-check-log-cli:
	pre-commit run detect-pytest-live-log --all-files

# Check format of all markdown files
lint-check-markdown:
	mdformat --check .

# Lint all files in the current directory (and any subdirectories).
lint: lint-python lint-toml lint-check-log-cli lint-check-markdown

# SOURCE: https://github.com/RobertCraigie/prisma-client-py/blob/da53c4280756f1a9bddc3407aa3b5f296aa8cc10/Makefile#L77
# Remove all generated files and caches
clean:
	rm -rf .cache
	rm -rf `find . -name __pycache__`
	rm -rf .tests_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf *.egg-info
	rm -f .coverage
	rm -f .coverage.*
	rm -rf build
	rm -rf dist
	rm -f coverage.xml

# generate type stubs for the project
createstubs:
	./scripts/createstubs.sh

# sweep init
sweep-init:
	sweep init

# Start a background HTTP server for test fixtures
http-server-background:
	#!/bin/bash
	# _PID=$(pgrep -f " -m http.server --bind localhost 19000 -d ./tests/fixtures")
	pkill -f " -m http.server --bind localhost 19000 -d ./tests/fixtures"
	python3 -m http.server --bind localhost 19000 -d ./tests/fixtures &
	echo $! > PATH.PID

# Start an HTTP server for test fixtures
http-server:
	#!/bin/bash
	# _PID=$(pgrep -f " -m http.server --bind localhost 19000 -d ./tests/fixtures")
	pkill -f " -m http.server --bind localhost 19000 -d ./tests/fixtures"
	python3 -m http.server --bind localhost 19000 -d ./tests/fixtures
	echo $! > PATH.PID

# Serve the documentation locally for preview
docs_preview:
	mkdocs serve

# Build the documentation
docs_build:
	mkdocs build

# Deploy the documentation to GitHub Pages
docs_deploy:
	mkdocs gh-deploy --clean

# Generate a draft changelog
changelog:
	towncrier build --version main --draft

# Checkout main branch and pull latest changes
gco:
	gco main
	git pull --rebase

# Show diff for LangChain migration
langchain-migrate-diff:
	langchain-cli migrate --include-ipynb --diff src

# Perform LangChain migration
langchain-migrate:
	langchain-cli migrate --include-ipynb src

get-ruff-config:
	ruff check --show-settings --config pyproject.toml -v -o ruff_config.toml >> ruff.log 2>&1

ci:
	lint
	test

manhole-shell:
	./scripts/manhole-shell

find-cassettes-dirs:
	fd -td cassettes

delete-cassettes:
	fd -td cassettes -X rm -ri

regenerate-cassettes:
	fd -td cassettes -X rm -ri
	unittests-vcr-record-final
	unittests-debug

brew-deps:
	brew install libmagic poppler tesseract pandoc qpdf tesseract-lang
	brew install --cask libreoffice

db-create:
	psql -d langchain -c 'CREATE EXTENSION vector'

typecheck:
	pyright -p pyproject.toml .
	mypy  --config-file=pyproject.toml --html-report typingcov .

install:
	pip install setuptools "cython >= 0.28"
	pip install -U -r requirements.txt
