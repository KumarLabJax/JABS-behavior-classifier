# JABS User Guide

Welcome to the JABS (JAX Animal Behavior System) user guide. This documentation will help you understand and use JABS for behavior classification in video data.

## What is JABS?

JABS is a machine learning-based tool for automatically identifying and classifying animal behaviors in video recordings. It uses pose estimation data to extract features and train classifiers that can recognize specific behaviors across multiple animals and videos.

**Key Features:**

- Train custom behavior classifiers with manual labeling
- Support for single and multi-animal pose estimation
- View prediction visualization during labeling to support active learning
- Batch processing via command-line tools
- Export classifiers for batch processing

## Before You Start

To use JABS effectively, you'll need:

- **Video files** of animal behavior (`.avi` or `.mp4` format)
- **Pose estimation files** corresponding to each video (from [Kumar Lab's pose estimation networks](https://github.com/KumarLabJax/deep-hrnet-mouse))
- **Time for labeling** - Quality classifiers require careful manual labeling of behavior examples
- **A behavior definition** - consistent labeling will help develop a better performing classifier


## Quick Start Guide

New to JABS? Follow these steps to get started:

1. **[Set up your project](project-setup.md#project-directory)** - Organize videos and pose files in a directory
2. **[Open the project](gui.md#main-window)** - Launch JABS and open your project directory
3. **[Add a behavior](gui.md#main-window)** - Create a new behavior to classify
4. **[Label examples](labeling.md)** - Select and label frames showing the behavior
5. **[Train a classifier](gui.md#classifier-controls)** - Train a model with your labeled data
6. **[Generate predictions](gui.md#classifier-controls)** - Apply the classifier to unlabeled frames
7. **[Search](behavior-search.md)** - Use Behavior Search to find and review labeled or predicted bouts

## Documentation Topics

Use the navigation tree on the left to explore different topics:

- **Project Setup** - Learn how to initialize and organize your JABS projects
- **GUI** - Understand the graphical user interface and its features
- **Labeling** - Master the labeling workflow for training classifiers
- **Searching Behaviors** - Find and review labeled or predicted behavior bouts
- **Command Line Tools** - Use JABS tools from the command line
- **File Formats** - Understand the format of JABS output files
- **Keyboard Shortcuts Reference** - Quick reference for all keyboard shortcuts

## Getting Help

### Need Assistance?

If you encounter issues or have questions about using JABS:

**📧 Email Support**

Contact the JABS team at **[jabs@jax.org](mailto:jabs@jax.org)** for:

- General questions about JABS functionality
- Help with installation or setup issues
- Questions about best practices for labeling and training

**🐛 Report Bugs or Issues**

Submit issues on our [GitHub repository](https://github.com/KumarLabJax/JABS-behavior-classifier/issues) for:

- Bug reports with detailed descriptions and error messages
- Feature requests or enhancement suggestions
- Documentation improvements

When reporting issues, please include:

- Your JABS version (shown in Help → About)
- Operating system and Python version
- Steps to reproduce the problem
- Any error messages or screenshots

### Additional Resources

- **[ReadTheDocs Tutorial](https://jabs-tutorial.readthedocs.io/en/latest/index.html)** - Comprehensive tutorials and examples
- **[GitHub Repository](https://github.com/KumarLabJax/JABS-behavior-classifier)** - Source code and latest releases
- **[Sample Data](https://doi.org/10.5281/zenodo.16697331)** - Download example datasets for testing

## Tips for Success

- **Start simple** - Begin with obvious, high-movement behaviors before tackling subtle ones
- **Label consistently** - Establish clear criteria for what constitutes the behavior
- **Use multiple animals** - Train on diverse examples to improve generalization
- **Balance your data** - Label both positive (behavior) and negative (not behavior) examples
- **Validate regularly** - Check predictions frequently to catch labeling mistakes early
- **Use keyboard shortcuts** - Master the hotkeys for efficient labeling (see [Keyboard Shortcuts](keyboard-shortcuts.md))
