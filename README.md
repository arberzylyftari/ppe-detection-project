# \---

# 

# \## Flask Webcam Application

# 

# A complete real-time detection web app built with Flask and YOLOv8.

# 

# \*\*Features:\*\*

# \- Live webcam detection with bounding boxes

# \- Upload any video file for batch detection

# \- Large SAFE / VIOLATION status indicator

# \- Live counters for all 9 classes

# \- FPS and inference time display

# \- Violations over time line chart

# \- Auto-screenshot on violation detection

# \- End-of-video Excel report export (3 sheets: Summary, Frame Log, Violation Timeline)

# \- Power BI compatible output

# 

# \*\*How to run:\*\*

# 

# ```bash

# git clone https://github.com/arberzylyftari/ppe-detection-yolov8.git

# cd ppe-detection-yolov8

# pip install -r requirements.txt

# \# Place best.pt (model weights) in this directory

# python app.py

# \# Open http://localhost:5000

# ```

# 

# > Note: `best.pt` model weights are not included in this repository due to file size.

# > Download from the project Google Drive or retrain using the notebook.

# 

# \---

# 

# \## Training Pipeline

# 

# Training was conducted across 5 experiments with increasing complexity:

# 

# | Experiment | Model | Classes | Images | mAP@0.5 |

# |---|---|---|---|---|

# | 1 | YOLOv8n | 3 | 7,040 | 66.0% |

# | 2 | YOLOv8n | 10 | \~15,000 | 53.8% |

# | 3 | YOLOv8s | 5 | \~28,000 | 89.8% |

# | 4 | YOLOv8s | 9 | \~45,000 | \~70.0% |

# | 5 | YOLOv8s | 9 | 82,459 | \*\*82.9%\*\* |

# 

# Full training code, results, and analysis in `PPE\_DetectionProject.ipynb`.

# 

# Training hardware: Google Colab (T4 GPU) and Kaggle (A100 GPU).

# 

# \---

# 

# \## Tech Stack

# 

# | Component | Technology |

# |---|---|

# | Detection model | YOLOv8 (Ultralytics) |

# | Backend | Flask |

# | Video processing | OpenCV |

# | Frontend | HTML5, JavaScript, Chart.js |

# | Report generation | openpyxl |

# | Reporting dashboard | Microsoft Power BI |

# | Dataset management | Roboflow |

# | Training environment | Google Colab, Kaggle |

# 

# \---

# 

# \## Known Limitations

# 

# \- Gloves and no-gloves detection (63%) is weaker due to small object size and occlusion

# \- Model trained on daytime images - night/low-light performance not evaluated

# \- Current deployment is single-machine; multi-camera setup requires server-side deployment

# 

# \---

# 

# \*ML and AI Final Project - 2026 - Arber Zylyftari\*

