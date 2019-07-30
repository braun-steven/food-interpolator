from PIL import Image
import sys
import os
import tqdm

# os.mkdir(sys.argv[1] + "-interp/")
def ensure_dir(path: str):
    """
    Ensure that a directory exists.

    Args:
        path (str): Directory path.
    """
    if not os.path.exists(path):
        os.makedirs(path)


base = sys.argv[1]
ensure_dir(base + "-scaled")
img_paths = list(sorted(filter(lambda x: x.endswith("png"), os.listdir(sys.argv[1]))))
print("Number of images:", len(img_paths))

for file in tqdm.tqdm(img_paths):
    try:
        im = Image.open(base + "/" + file)
        new_width, new_height = 2066, 1034
        im_resized = im.resize((new_width, new_height))
        im_resized.save(base + "-scaled/" + file)
    except Exception as e:
        print("Exception for file:", file)
        print(e)
        print("Skipping...")
