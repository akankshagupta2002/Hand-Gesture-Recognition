\# Hand Gesture Recognition



A computer vision project for recognizing hand gestures from images and real-time webcam frames using MediaPipe hand landmarks and machine learning.



Developed as part of an AI/ML Internship — Task 5: Hand Gesture Recognition.



\---



\## Project Overview



This project recognizes 10 different hand gestures using a two-stage computer vision pipeline:



1\. MediaPipe Hand Landmarker detects the hand and extracts 21 hand landmarks.

2\. A Random Forest classifier predicts the static hand gesture from the normalized landmark coordinates.

3\. Temporal motion detection distinguishes moving gestures such as Palm Moved and Fist Moved.

4\. Prediction smoothing is applied to make real-time predictions more stable.



The project also includes a MobileNetV2 model trained on the LeapGestRecog dataset for image-based evaluation and an uploaded-image fallback for IR-style dataset images.



\---



\## Dataset



The project uses the \*\*LeapGestRecog\*\* hand gesture dataset.



Dataset source:



https://www.kaggle.com/datasets/gti-upm/leapgestrecog



The dataset contains 10 gesture classes:

| Class | Gesture |
|---|---|
| `01_palm` | Palm |
| `02_l` | L |
| `03_fist` | Fist |
| `04_fist_moved` | Fist Moved |
| `05_thumb` | Thumb |
| `06_index` | Index |
| `07_ok` | OK |
| `08_palm_moved` | Palm Moved |
| `09_c` | C |
| `10_down` | Down |




\### Dataset Split



For the primary MobileNetV2 evaluation, the data was split subject-wise to avoid subject leakage:



\- Training subjects: 00–07

\- Validation subject: 08

\- Test subject: 09

\- Training images: 16,000

\- Validation images: 2,000

\- Test images: 2,000

\- Total images: 20,000



The subject-wise split provides a more meaningful test of generalization to an unseen subject.



\---



\## Project Pipeline



```text

Input Image / Webcam Frame

&#x20;         |

&#x20;         v

MediaPipe Hand Detection

&#x20;         |

&#x20;         v

21 Hand Landmarks

&#x20;         |

&#x20;         v

Landmark Normalization

&#x20;         |

&#x20;         v

Random Forest Classifier

&#x20;         |

&#x20;         v

Gesture Prediction

&#x20;         |

&#x20;         +------> Motion Detection

&#x20;         |

&#x20;         +------> Prediction Smoothing

&#x20;         |

&#x20;         v

Final Gesture

