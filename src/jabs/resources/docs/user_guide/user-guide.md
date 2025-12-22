# JABS User Guide

Welcome to the JABS (JAX Animal Behavior System) user guide. This documentation will help you understand and use JABS for behavior classification in video data.

## Table of Contents

### [Overview](overview.md)
Quick introduction and overview of JABS documentation structure.

### [Project Setup](project-setup.md)
Learn how to initialize and organize your JABS projects:
- [Project Directory](project-setup.md#project-directory) - Understanding JABS project structure
- [Initialization & jabs-init](project-setup.md#initialization-jabs-init) - Using the jabs-init script to prepare projects
- [JABS Directory Structure](project-setup.md#jabs-directory-structure) - Understanding the jabs/ subdirectory contents

### [GUI](gui.md)
Understand the graphical user interface and its features:
- [Main Window](gui.md#main-window) - Overview of the JABS main interface
- [Classifier Controls](gui.md#classifier-controls) - Training and configuring classifiers
- [Timeline Visualizations](gui.md#timeline-visualizations) - Understanding label and prediction displays
- [Video Controls](gui.md#video-controls) - Video playback and manipulation controls
- [Menu](gui.md#menu) - Menu options and features
- [Overlays](gui.md#overlays) - Track, pose, and identity overlay options

### [Labeling](labeling.md)
Master the labeling workflow for training classifiers:
- [Selecting Frames](labeling.md#selecting-frames) - How to select frames for labeling
- [Applying Labels](labeling.md#applying-labels) - Applying behavior and non-behavior labels
- [Timeline Annotations](labeling.md#timeline-annotations) - Using timeline annotations to mark special frames
- [Identity Gaps](labeling.md#identity-gaps) - Understanding and handling identity gaps
- [Keyboard Shortcuts](labeling.md#keyboard-shortcuts) - Labeling-specific keyboard shortcuts

### [Command Line Tools](cli-tools.md)
Use JABS tools from the command line:
- [jabs-classify](cli-tools.md#jabs-classify) - Command line classifier for batch processing
- [jabs-features](cli-tools.md#jabs-features) - Feature generation from the command line
- [jabs-cli](cli-tools.md#jabs-cli) - Additional command line utilities

### [File Formats](file-formats.md)
Understand the format of JABS output files:
- [Prediction File](file-formats.md#prediction-file) - Format of classifier prediction outputs
- [Feature File](file-formats.md#feature-file) - Format of computed feature files

### [Keyboard Shortcuts Reference](keyboard-shortcuts.md)
Complete reference of all keyboard shortcuts available in JABS.

---

## Additional Resources

- **ReadTheDocs Tutorial**: https://jabs-tutorial.readthedocs.io/en/latest/index.html
- **GitHub Repository**: https://github.com/KumarLabJax/JABS-behavior-classifier
- **Contact**: jabs@jax.org
