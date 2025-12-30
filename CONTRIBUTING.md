# Contributing to JABS

Thank you for your interest in contributing to JABS (JAX Animal Behavior System)! This document provides guidelines for contributing to the project.

## Copyright and License

**Important:** By contributing source code to JABS, you agree that your contributions will have their copyright assigned to The Jackson Laboratory.

All code contributions become part of the JABS project and are subject to the project's license terms (see the LICENSE file in the repository root). JABS is licensed under a non-commercial use license. Contact jabs@jax.org for information about commercial licensing.

## Quick Start for Contributors

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/JABS-behavior-classifier.git
cd JABS-behavior-classifier

# 2. Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS/Linux
# OR: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# 3. Set up development environment
uv sync
source .venv/bin/activate  # macOS/Linux (or .venv\Scripts\activate on Windows)
pre-commit install

# 4. Create a feature branch
git checkout -b feature/my-feature

# 5. Make your changes and commit
git add .
git commit -m "Description of changes"

# 6. Run tests
pytest

# 7. Push and create a pull request
git push origin feature/my-feature
```

## Contribution Guidelines

Before submitting a contribution:

1. **Follow the code style guidelines** - JABS uses Ruff for linting and formatting
2. **Add tests** for new functionality
3. **Update documentation** as needed (docstrings, user guide, developer guide)
4. **Run the test suite** to ensure nothing is broken: `pytest`
5. **Ensure pre-commit hooks pass** - They will run automatically when you commit
6. **Submit a pull request** with a clear description of your changes

### Code Style

JABS uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

Pre-commit hooks will automatically run these checks before each commit. If your commit is blocked:

1. Fix issues: `ruff check --fix . && ruff format .`
2. Stage fixes: `git add -u`
3. Try committing again

### Running Tests

```bash
pytest              # Run all tests
pytest -v           # Verbose output
pytest tests/path/  # Run specific tests
```

## Types of Contributions Welcome

We welcome various types of contributions:

- **Bug fixes**: Help identify and fix issues
- **New features**: Add new behavioral features, classifiers, or GUI improvements
- **Documentation**: Improve user guides, developer documentation, or code comments
- **Testing**: Add test coverage or improve existing tests
- **Performance improvements**: Optimize code for better performance

## Developer Documentation

For detailed information about JABS architecture, development setup, and implementation guides, see:

- **[Development Guide](docs/DEVELOPMENT.md)** - Comprehensive guide to JABS architecture, feature extraction system, building, and deployment
- **[User Guide](docs/user-guide.md)** - End-user documentation

## Pull Request Process

1. **Create a feature branch** from `main`
2. **Make your changes** with clear, focused commits
3. **Update tests** and documentation
4. **Ensure all tests pass** locally
5. **Create a pull request** against the `main` branch
6. **Wait for review** - Maintainers will review and may request changes
7. **CI/CD checks must pass** - Automated checks will run on your PR

## Questions?

- **General questions**: Open an issue on GitHub
- **Security issues**: Email jabs@jax.org (do not open public issues)
- **Development questions**: See [DEVELOPMENT.md](docs/DEVELOPMENT.md) or contact jabs@jax.org

## Code of Conduct

Be respectful and professional in all interactions. We aim to foster an inclusive and welcoming community.

---

**Thank you for contributing to JABS!**

