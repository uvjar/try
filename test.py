import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np
import string

characters = ' '+ string.digits + string.ascii_lowercase+string.punctuation


def my_ctc_decode(y_pred):
  decode =(np.ones((1,16))*(-1)).astype('int64')
  idx=0;prev=69;
  for i in range(16):
    t=np.argmax(y_pred[0][i])
    if t!=69:
      if t!=prev:
        decode[0][idx]=t;idx=idx+1;
    prev=t
  return decode


def main():
	interpreter = tflite.Interpreter(model_path="converted_model.tflite")
	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	input_shape = input_details[0]['shape']

	for i in range(2):
		file = 'test'+str(i)+'.png';ff = Image.open(file);dd = np.array(ff).astype('float32');
		dd = np.array([dd/255.0])
		interpreter.set_tensor(input_details[0]['index'], dd)
		interpreter.invoke()
		y_pred = interpreter.get_tensor(output_details[0]['index'])
		out=my_ctc_decode(y_pred)[:,:6];print(out)
		out = ''.join([characters[x] for x in out[0]])
		print(file+" "+out)

  




if __name__ == '__main__':
	main()