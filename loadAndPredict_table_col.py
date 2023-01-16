import pytesseract
import tensorflow as tf
from trainModel_table_col import build_model, checkpoint_dir, test_dataset_dir
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def main():
    model = build_model()
    # test_dataset = tf.data.Dataset.load(test_dataset_dir)
    # loading trained model
    latest_cp = tf.train.latest_checkpoint(checkpoint_dir)
    # expect_partial() unterdrückt warnungen, dass nicht alle gespeicherten variablen der checkpoints genutzt werden
    # dies hat den hintergrund, dass bei model.predict() die variablen des optimizers für evtl. weitere Datensätze
    # nicht mehr benötigt werden
    model.load_weights(latest_cp).expect_partial()
    images = os.listdir(os.path.join("PDF_Out"))
    r = np.random.randint(len(images))
    img_fname = os.path.join("PDF_Out", "januar20.jpg")
    print(img_fname)
    masked_image = create_masked_image(img_fname, model)
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    #plt.imshow(masked_image, cmap="gray")
    #plt.show()
    get_table(masked_image)


# ref: https://www.tensorflow.org/tutorials/images/segmentation
def create_masks(msk):
    tf_msk = tf.argmax(msk, axis=3)
    tf_msk = tf_msk[..., tf.newaxis]
    return tf_msk[0]


# ref: https://stackoverflow.com/questions/10469235/opencv-apply-mask-to-a-color-image
def create_masked_image(image_fname, model):
    dim = (1024, 1024)
    orig_dim = Image.open(image_fname).size
    decoded = tf.io.decode_jpeg(tf.io.read_file(image_fname), channels=3)
    decoded = tf.expand_dims(decoded, axis=0)  # Network expects batches, so we're giving it a batch with a single image
    img = tf.cast(decoded, tf.float32)
    img = tf.image.resize(img, dim)
    img = tf.cast(img, tf.float32) / 255
    t_mask, c_mask = model.predict(img)
    t_mask, c_mask = create_masks(t_mask), create_masks(c_mask)

    t_mask_img = tf.keras.utils.array_to_img(t_mask).resize(orig_dim)
    c_mask_img = tf.keras.utils.array_to_img(c_mask).resize(orig_dim)

    t_mask_arr = np.array(t_mask_img)
    c_mask_arr = np.array(c_mask_img)
    orig_image_array = np.array(Image.open(image_fname))

    # logic AND to combine masks
    combined_mask = t_mask_arr & c_mask_arr
    masked_img = cv2.bitwise_and(orig_image_array, orig_image_array, mask=combined_mask)

    image_list = [orig_image_array, t_mask_img, c_mask_img, masked_img]
    plt.figure(figsize=(15, 10))
    names = ["Input Image", "Table Mask", "Column Mask", "Combined Masked Image"]
    for i in range(len(image_list)):
        plt.subplot(2, 2, i + 1)  # Adding 1, because pyplot only accepts indices greater than 0
        plt.title(names[i])
        plt.imshow(image_list[i])
        plt.axis('off')
    plt.show()

    return masked_img


def get_table(masked_img):
    tessdata_dir_config = '--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata" --psm 13'
    text = pytesseract.image_to_string(masked_img, lang="deu", config=tessdata_dir_config)
    text = text
    print(text)


if __name__ == "__main__":
    main()
