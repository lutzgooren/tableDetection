import pytesseract
import tensorflow as tf
from trainModel_head_front_field import build_model, checkpoint_dir, test_dataset_dir
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def main():
    model = build_model()
    # test_dataset = tf.data.Dataset.load(test_dataset_dir)
    # lade trainiertes model
    latest_cp = tf.train.latest_checkpoint(checkpoint_dir)
    # expect_partial() unterdrückt warnungen, dass nicht alle gespeicherten variablen der checkpoints genutzt werden
    # dies hat den hintergrund, dass bei model.predict() die variablen des optimizers für evtl. weitere Datensätze
    # nicht mehr benötigt werden
    model.load_weights(latest_cp).expect_partial()
    images = os.listdir(os.path.join("Berlin2020"))
    r = np.random.randint(len(images))
    img_fname = os.path.join("Berlin2020", "januar20-3.jpg")
    print(img_fname)
    masked_image = create_masked_image(img_fname, model)
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
    # Netzwerk erwartet batches, also wird durch expand_dims() eine batch mit einem einzigen bild übergeben
    decoded = tf.expand_dims(decoded, axis=0)
    img = tf.cast(decoded, tf.float32)
    img = tf.image.resize(img, dim)
    img = tf.cast(img, tf.float32) / 255
    th_mask, fc_mask, f_mask = model.predict(img)
    th_mask, fc_mask, f_mask = create_masks(th_mask), create_masks(fc_mask), create_masks(f_mask)

    # konvertiere vorhersagen zu bildern und skaliere
    th_mask_img = tf.keras.utils.array_to_img(th_mask).resize(orig_dim)
    fc_mask_img = tf.keras.utils.array_to_img(fc_mask).resize(orig_dim)
    f_mask_img = tf.keras.utils.array_to_img(f_mask).resize(orig_dim)

    th_mask_arr = np.array(th_mask_img)
    fc_mask_arr = np.array(fc_mask_img)
    f_mask_arr = np.array(f_mask_img)
    orig_image_array = np.array(Image.open(image_fname))

    combined_mask = th_mask_arr | fc_mask_arr | f_mask_arr
    masked_img = cv2.bitwise_and(orig_image_array, orig_image_array, mask=combined_mask)

    image_list = [orig_image_array, th_mask_img, fc_mask_img, f_mask_img, masked_img]
    plt.figure(figsize=(15, 10))
    names = ["Input Image", "Tablehead Mask", "Frontcolumn Mask", "Field Mask", "Combined Masked Image"]
    for i in range(len(image_list)):
        plt.subplot(2, 3, i + 1)  # +1, da pyplot nur indizes größer 0 unterstützt
        plt.title(names[i])
        plt.imshow(image_list[i])
        plt.axis('off')
    plt.show()

    return masked_img


def get_table(masked_img):
    tessdata_dir_config = '--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"'
    text = pytesseract.image_to_string(masked_img, lang="deu", config=tessdata_dir_config)
    text = text
    print(text)


if __name__ == "__main__":
    main()
