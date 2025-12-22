# GUI

## Main Window

<img src="imgs/main_window.png" alt="JABS Main Window" width=1150 />

- **Behavior Selection:** Select current behavior to label
- **Add New Behavior Button:** Add new behavior label to project
- **Identity Selection:** Select subject mouse to label (subject can also be selected by clicking on mouse in the video)
- **Classifier Controls:** Configure and train classifier. Use trained classifier to infer classes for unlabeled frames. See "Classifier Controls" section for more details.
- **Label Summary:** Counts of labeled frames and bouts for the subject identity in the current video and across the whole project.
- **Label "Behavior" Button:** Label current selection of frames as showing behavior. This button is labeled with the current behavior name.
- **Label "Not Behavior" Button:** Label current selection of frames as not showing behavior. This button is labeled with `Not <current behavior name>`.
- **Clear Selection Button:** remove labels from current selection of frames
- **Toggle Select Mode Button:** toggle select mode on/off (turning select mode on will begin selecting frames starting from that point)
- **Video Playlist:** list of videos in the current project. Click a video name to make it the active video.
- **Video Player:** Displays the current video. See "Video Player" section for more information.
- **Manual Label and Predicted Label Visualizations:** see "Label Visualizations" for more information.
- **Status Bar:** Displays periodic status messages.

## Classifier Controls

<img src="imgs/classifier_controls.png" alt="JABS Classifier Controls" width=900 />

- **Train Button:** Train the classifier with the current parameters. This button is disabled until minimum number of frames have been labeled for a minimum number of mice (increasing the cross validation k parameter increases the minimum number of labeled mice)
- **Classify Button:** Infer class of unlabeled frames. Disabled until classifier is trained. Changing classifier parameters may require retraining before the Classify button becomes active again.
- **Classifier Type Selection:** Users can select from a list of supported classifiers.
- **Window Size Selection:** Number of frames on each side of the current frame to include in window feature calculations for that frame. A "window size" of 5 means that 11 frames are included into the window feature calculations for each frame (5 previous frames, current frame, 5 following frames).
- **New Window Size:** Add a new window size to the project.
- **Label Balancing Toggle:** Balances the training data by downsampling the class with more labels such that the distribution is equal.
- **Symmetric Behavior Toggle:** Tells the classifier that the behavior is symmetric. A symmetric behavior is when left and right features are interchangeable.
- **All k-fold Toggle:** Uses the maximum number of cross validation folds. Useful when you wish to compare classifier performance and may have an outlier that can be held-out.
- **Cross Validation Slider:** Number of "Leave One Out" cross validation iterations to run while training.

## Timeline Visualizations

<img src="imgs/label_viz.png" alt="JABS Label Visualizations" width=900 />

- **Manual Labels (sliding window):** Displays manually assigned labels for a sliding window of frames. The window range is the current frame +/-50 frames. Orange indicates frames labeled as showing the behavior, blue indicates frames labeled as not showing the behavior. Unlabeled frames are colored gray.
- **Manual Labels (global view):** Displays a zoomed out view of the manual labels for the entire video
- **Predicted Classes (sliding window):** Displays predicted classes (if the classifier has been run). Color opacity indicates prediction probability for the predicted class. Manually assigned labels are also displayed with probability of 100%.
- **Predicted Class (global view):** Displays a zoomed out view of the predicted classes for the entire video.
- **Sliding Window Indicator:** highlights the section of the global views that correspond to the frames displayed in the "sliding window" views.

By default, the Timeline shows manual labels and predicted behaviors for the current subject animal. The Timeline can be toggled to show all subjects by selecting View->Timeline->All Animals in the menu bar. The Timeline can also be configured to show only manual labels or only predicted labels. If "All Animals" is selected, the Timeline will show which set of labels and predictions belong to the subject animal by drawing a colored border around them.

### Timeline Menu

<img src="imgs/timeline_menu.png" alt="Timeline visualization options" />

<br /><br />

### Example Timeline with "Labels & Predictions" and "All Animals" selected

<img src="imgs/stacked_timeline.png" alt="Timeline with all animals" width=900 />

## Video Controls

<img src="imgs/video-control-overlay.png" alt="Video Control Overlay" width=900 />

Mousing over the video player will display a control overlay with the following controls:

- **Video Playback Speed Controls:** Controls the speed of video playback. Clicking this control will open a menu with options for playback speed. The default speed is 1x.
- **Video Cropping:** Allows the user to crop the video to a specific region of interest. After clicking the cropping control, the user can click and drag a rectangular selecting tool to select the region of interest. The video will be cropped to the selection, and scaled to fill the available player area. If the video is currently cropped, the cropping control will be replaced with a "Reset Cropping" control, which will reset the cropping to the original video size.
- **Brightness Adjustment:** Allows the user to adjust the brightness of the video. Clicking this control will open a slider that can be used to adjust the brightness.
- **Contrast Adjustment:** Allows the user to adjust the contrast of the video. Clicking this control will open a slider that can be used to adjust the contrast.

Clicking the Brightness or Contrast controls will reset the brightness or contrast to the default value before displaying the slider control. Clicking the video or moving the mouse off the video frame will dismiss the slider control.

## Menu

- **JABS→About:** Display About Dialog
- **JABS→User Guide:** Display User Guide
- **JABS→Quit JABS:** Quit Program
- **File→Open Project:** Select a project directory to open. If a project is already opened, it will be closed and the newly selected project will be opened.
- **File→Export Training Data:** Create a file with the information needed to share a classifier. This exported file is written to the project directory and has the form `<Behavior_Name>_training_<YYYYMMDD_hhmmss>.h5`. This file is used as one input for the `jabs-classify` script.
- **File→Archive Behavior:** Remove behavior and its labels from project. Labels are archived in the `jabs/archive` directory.
- **File→Prune Project:** Remove videos and pose files that are not labeled.
- **View:** Menu to control various display options.
  - **View→View Playlist:** can be used to hide/show video playlist
  - **View→Timeline:** Menu to control the timeline display.
  - **View→Label Overlay:** Control the floating display of manual labels or predicted classes.
  - **View→Identity Overlay:** Configure the identity overlay mode.
  - **View→Show Track:** show/hide track overlay for the subject. The track overlay shows the nose position for the previous 5 frames and the next 10 frames. The nose position for the next 10 frames is colored red, and the previous 5 frames it is a shade of pink.
  - **View→Overlay Pose:** toggle the overlay of the pose on top of the subject mouse
  - **View→Overlay Landmarks:** toggle the overlay of arena landmarks over the video.
- **Features:** Menu item for controlling per-behavior classifier settings. Menu items are disabled when at least 1 pose file in the project does not contain the data to calculate features.
  - **Features→CM Units:** toggle using CM or pixel units (Warning! Changing this will require features to be re-calculated)
  - **Features→Enable Window Features:** toggle using statistical window features
  - **Features→Enable Signal Features:** toggle using fft-based window features
  - **Features→Enable Social Features:** toggle using social features (v3+ projects)
  - **Features→Enable Corners Features:** toggle using arena corner features (v5+ projects with arena corner static object)
  - **Features→Enable Lixit Features:** toggle using lixit features (v5+ projects with lixit static object)
  - **Features→Enable Food_hopper Features:** toggle using food hopper features (v5+ projects with food hopper static object)
  - **Features→Enable Segmentation Features:** toggle using segmentation features (v6+ projects)

## Overlays

### Track Overlay Example

<img src="imgs/track_overlay.png" alt="Track Overlay" width=400 />

### Pose Overlay Example

<img src="imgs/pose_overlay.png" alt="Pose Overlay" />

### Pose Overlay Keypoint Legend

<img src="imgs/keypoint_legend.png" alt="Pose Keypoint Legend" width="500"/>

### Identity Overlay

JABS offers several ways to overlay mouse identities on the video. Choose a mode from View → Identity Overlay.

In all modes, you can select the subject directly in the video: click inside the convex hull of body keypoints (excluding the tail) to select that animal. You can also click the floating identity label. With the Bounding Box overlay, clicking the tab selects the animal. These options are in addition to the Identity dropdown in the main window and the Shift+↑ / Shift+↓ keyboard shortcuts.

#### Floating

<img src="imgs/floating-identity-overlay.png" alt="Floating Identity Overlay" />

#### Centroid

<img src="imgs/centroid-identity-overlay.png" alt="Centroid Identity Overlay" />

#### Minimalist

<img src="imgs/minimalist-identity-overlay.png" alt="Minimalist Identity Overlay" />

#### Bounding Box

<img src="imgs/bbox-identity-overlay.png" alt="Bounding Box Identity Overlay" width="600"/>
