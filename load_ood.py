from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

def load_image( infilename ) :
    try:
        img = Image.open( infilename )
        img.load()
        img = img.resize((32, 32))
        data = np.asarray( img, dtype="int32" )
        data = data/255
        return data
    except Exception:
        return None


def progressbar(i, maxsize, pre_text):
    n_bar = 10
    j = i / maxsize
    print('\r', end="")
    print(f"{pre_text} [{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%", end="", flush=True)


def getImages(folder_dir, img_type = ".jpg"):
	images = []
	files = Path(folder_dir).glob(f'**/*{img_type}')
	files_size = len(list(files))
	count = 1
	print(str(files_size) + "\n")
	for image_path in Path(folder_dir).glob(f'**/*{img_type}'):
		data = load_image(image_path)
		if data is None:
			continue
		data = data.flatten()
		if len(data) == 3072:
			images.append(data)
		progressbar(count, files_size, "Getting Images")
		count+=1
	return np.array(images)

def start(folder_dir):
	images = getImages(folder_dir, ".jpg")
	imgs2 = getImages(folder_dir, ".png")
	if len(imgs2) > 0:
		print("PNGS: ", imgs2.shape)
		images = images + imgs2
	imgs3 = getImages(folder_dir, ".jpeg")
	if len(imgs3) > 0:
		print("JPEGS: ", imgs3.shape)
		images = images + imgs3

	df = pd.DataFrame(images, columns = ["data"]*3072)
	df.to_pickle(f"{folder_dir}.pkl")

def main():
	start("CUB")
	start("food")
	start("stanford_dogs")
	start("Calt")
	start("dtd")
	start("places")

main()
