@echo off
python -c "import OpenEXR, customtkinter, cv2, PIL, numpy" 2>NUL
if errorlevel 1 (
    echo Installing required Python packages ^(OpenEXR v3.5.3, CustomTkinter, OpenCV^)...
    pip install -r requirements.txt
)
python indigo_oidn_gui.py
pause