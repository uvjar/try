## create virtual environment
python3 -m venv scp2/
source scp2/bin/activate

# install packages
pip instal numpy pillow

# install tflite runtime
git clone https://github.com/uvjar/try.git
pip install try/tflite_runtime-2.5.0-cp38-cp38-linux_armv7l.whl

## Exceute on PI
python3 try/test.py --model try/converted_model.tflite --inputDir try/CAOP-project2rpi/ --output cpy_live.txt

