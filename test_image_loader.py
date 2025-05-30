from src.utils import load_image, preprocess_image
import matplotlib.pyplot as plt

image = load_image("img_folder", "sample1")
processed = preprocess_image(image)

plt.imshow(image)
plt.title("Loaded Image")
plt.axis("off")
plt.show()
