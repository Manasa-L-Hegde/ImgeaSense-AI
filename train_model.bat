@echo off
echo ===================================================
echo Training ImageSense AI Model on Flickr8k
echo ===================================================
"C:\Users\manas\AppData\Local\Programs\Python\Python312\python.exe" -m src.train --data-dir data --max-images 300 --epochs 5 --batch-size 32
pause

