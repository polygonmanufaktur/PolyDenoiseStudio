@echo off
setlocal
pushd "%~dp0"

python -c "import OpenEXR, customtkinter, cv2, PIL, numpy, moderngl" 2>NUL
if errorlevel 1 (
    echo Installing required Python packages ^(OpenEXR, CustomTkinter, ModernGL, OpenCV^)...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        popd
        pause
        exit /b 1
    )
)

python -c "from polydenoise import constants, filter_registry" 2>NUL
if errorlevel 1 (
    echo [ERROR] Failed to import modular 'polydenoise' package from "%~dp0".
    popd
    pause
    exit /b 1
)

python -m polydenoise %*
if errorlevel 1 (
    echo.
    echo [ERROR] PolyDenoise Studio exited with an error.
    pause
)

popd
endlocal
