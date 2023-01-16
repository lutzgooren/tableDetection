import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Layer, Conv2D, UpSampling2D, Concatenate, Conv2DTranspose, Dropout
from keras.applications import VGG16, VGG19, ResNet50, DenseNet201, EfficientNetB5, Xception, NASNetLarge
from keras import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils.vis_utils import plot_model

# setze verzeichnis der trainingsdaten und erstelle dataframe aus zuvor ersteller .csv datei
train_dir = os.path.join("stuff")
test_dataset_dir = "D:/datasets/test_data"
checkpoint_path = "D:/keras_models/table_col/learning_1e-03_epsilon_1e-08_dropout_0.8/VGG19_1024/cp_{epoch:04d}_{val_loss:.4f}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
log_dir = "D:/keras_models/table_col/learning_1e-03_epsilon_1e-08_dropout_0.8/VGG19_1024_logs"


def main():
    # erstelle dataframe aus zuvor ersteller .csv datei
    train_df = pd.read_csv(os.path.join(train_dir, "csv", "data.csv"))

    # setzt das backend zurück - nur zur sicherheit
    tf.keras.backend.clear_session()
    # from_tensor_slices erstellt jeweils ein element pro reihe des datensatzes
    dataset = tf.data.Dataset.from_tensor_slices((train_df['image_path'].values,
                                                  train_df["tablemask_path"],
                                                  train_df["columnmask_path"]))

    # erstellt train, validation und test datensatz
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = int(0.1 * dataset_size)

    dataset = dataset.shuffle(buffer_size=42)
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    val_dataset = test_dataset.skip(test_size)
    test_dataset = test_dataset.take(test_size)

    train_dataset = train_dataset.map(map_function).batch(1)
    val_dataset = val_dataset.map(map_function).batch(1)
    test_dataset = test_dataset.map(map_function).batch(1)

    if not os.path.exists(test_dataset_dir):
        os.makedirs(test_dataset_dir)
    test_dataset.save(test_dataset_dir)

    model = build_model()
    model.summary()

    # gibt ein "flow-chart" des zuvor erstellten models aus
    plot_model(model, to_file="model_plot.png", show_shapes=True, show_layer_names=True)

    # erstelle metriken für den verlust
    losses = {
        "table": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        "columns": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    }

    # möglichkeit, um die gewichtung der verluste anzupassen - standard 1.0 für alle
    lossWeights = {
        "table": 1.0,
        "columns": 1.0
    }

    # einstellen des adam algorithmus für das neuronale netz
    optim = tf.keras.optimizers.Adam(learning_rate=1e-03, epsilon=1e-08)

    # erstelle score für die genauigkeit (sc_acc)
    sc_acc = tf.keras.metrics.SparseCategoricalAccuracy(name="sc_accuracy")

    # kompiliert das model mit den zuvor eingestellten parametern
    model.compile(optimizer=optim,
                  loss=losses,
                  metrics=[sc_acc],
                  loss_weights=lossWeights)

    # zeige vorhersagen, bevor das netz trainiert worden ist
    # ref: https://www.tensorflow.org/tutorials/images/segmentation
    """
    for image, msks in val_dataset.take(1):
        table_mask, column_mask = msks['table'], msks['columns']
        pred_tab_mask, pred_c_mask = model.predict(image)
        show_imgs([image[0], create_masks(pred_tab_mask), create_masks(pred_c_mask)])
    """

    # erstelle model callbacks für checkpoints und tensorboard für spätere analyse der logs
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # checkpoints werden bei verbesserter validation loss gespeichert
    cp_callback = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True, monitor='val_loss',
                                  save_weights_only=True)
    # tensorboard callback für des spätere auswerten der logs
    tb_callback = TensorBoard(log_dir)
    callbacks = [cp_callback, tb_callback]

    model.fit(
        train_dataset, epochs=200,
        batch_size=len(train_dataset) // 2,
        steps_per_epoch=len(train_dataset) // 1,
        validation_data=val_dataset,
        validation_steps=len(val_dataset) // 1,
        callbacks=callbacks
    )


# funktion um eine liste von bilder anzeigen zu lassen
def show_imgs(image_list):
    plt.figure(figsize=(15, 10))
    names = ["Input Image", "Table Mask", "Column Mask"]
    for i in range(len(image_list)):
        plt.subplot(1, len(image_list), i + 1)  # +1, da pyplot nur indizes größer 0 unterstützt
        plt.title(names[i])
        plt.imshow(tf.keras.utils.array_to_img(image_list[i]))
        plt.axis('off')
    plt.show()


# vorbereitung der trainingsdaten durch map_function()
def map_function(img, t_mask, col_mask):
    """
    nimmt jpg dateien aus dem dataset entgegen und dekodiert diese in ein für tensorflow brauchbares format
    das bild selbst wird in RGB dekodiert - die masken werden mit nur einem channel dekodiert
    alle bilder werden auf eine zuvor festgelegte größe (dim) skaliert und die farben werden von 0 bis 1 normalisiert
    """
    dim = (1024, 1024)
    decoded = tf.io.decode_jpeg(tf.io.read_file(img), channels=3)  # image in RGB
    img = tf.cast(decoded, tf.float32)
    img = tf.image.resize(img, dim)
    img = tf.cast(img, tf.float32) / 255

    t_msk_decoded = tf.io.decode_jpeg(tf.io.read_file(t_mask), channels=1)
    t_msk = tf.cast(t_msk_decoded, tf.float32)
    t_msk = tf.image.resize(t_msk, dim)
    t_msk = tf.cast(t_msk, tf.float32) / 255

    col_msk_decoded = tf.io.decode_jpeg(tf.io.read_file(col_mask), channels=1)
    c_msk = tf.cast(col_msk_decoded, tf.float32)
    c_msk = tf.image.resize(c_msk, dim)
    c_msk = tf.cast(c_msk, tf.float32) / 255

    mask_dict = {
        'table': t_msk,
        'columns': c_msk
    }

    return img, mask_dict


class TableConvLayer(Layer):
    """
    ref: TableNet: Deep Learning model for end-to-end
    Table detection and Tabular data extraction from
    Scanned Document Images
    Shubham Paliwal, Vishwanath D, Rohit Rahul, Monika Sharma, Lovekesh Vig
    TCS Research, New Delhi
    {shubham.p3, vishwanath.d2, monika.sharma1, rohit.rahul, lovekesh.vig}@tcs.com
    Creating new custom layer classes for neural network
    ref: https://faroit.com/keras-docs/2.0.1/layers/writing-your-own-keras-layers/#writing-your-own-keras-layers
    """
    def __int__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(TableConvLayer, self).__int__(**kwargs)

    def build(self, input_shape):
        self.conv7 = Conv2D(512, 1, activation='relu', name='conv7table')
        self.upsample_conv7 = UpSampling2D(2)
        self.concat_p4 = Concatenate()
        self.upsample_p4 = UpSampling2D(2)

        self.concat_p3 = Concatenate()
        self.upsample_p3 = UpSampling2D(2)

        self.upsample_p3_2 = UpSampling2D(2)
        self.convtranspose = Conv2DTranspose(3, 3, strides=2, padding='same')
        super(TableConvLayer, self).build(input_shape)

    def call(self, x):
        x, y, z = x
        x = self.conv7(x)
        x = self.upsample_conv7(x)
        x = self.concat_p4([x, z])
        x = self.upsample_p4(x)
        x = self.concat_p3([x, y])
        x = self.upsample_p3(x)
        x = self.upsample_p3_2(x)
        x = self.convtranspose(x)

        return x


class ColumnConvLayer(Layer):

    def __int__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(ColumnConvLayer, self).__int__(**kwargs)

    def build(self, input_shape):
        self.conv7 = Conv2D(512, 1, activation='relu', name='conv7columns')
        self.upsample_conv7 = UpSampling2D(2)
        self.concat_p4 = Concatenate()
        self.upsample_p4 = UpSampling2D(2)

        self.concat_p3 = Concatenate()
        self.upsample_p3 = UpSampling2D(2)

        self.upsample_p3_2 = UpSampling2D(2)
        self.convtranspose = Conv2DTranspose(3, 3, strides=2, padding='same')
        super(ColumnConvLayer, self).build(input_shape)

    def call(self, x):
        x, y, z = x
        x = self.conv7(x)
        x = self.upsample_conv7(x)
        x = self.concat_p4([x, z])
        x = self.upsample_p4(x)
        x = self.concat_p3([x, y])
        x = self.upsample_p3(x)
        x = self.upsample_p3_2(x)
        x = self.convtranspose(x)

        return x


def build_model():
    """
    erstellt das model mit hilfe eines bereits existierenden netzes und dem anschließenden anhängen der zuvor
    erstellten output layer
    input shape kann je nach verwendetem netz angepasst werden, um in vorhandenen speicher zu passen
    """
    tf.keras.backend.clear_session()
    input_shape = (1024, 1024, 3)

    base = VGG19(input_shape=input_shape, include_top=False, weights='imagenet')

    # VGG 16/19
    end_layers_list = ['block3_pool', 'block4_pool', 'block5_pool']

    # ResNet50
    # end_layers_list = ['conv3_block4_3_conv', 'conv4_block6_3_conv', 'conv5_block3_3_conv']

    # DenseNet201
    # end_layers_list = ['pool3_pool', 'pool4_pool', 'relu']

    # Xception
    # end_layers_list = ['block10_sepconv1', 'block12_sepconv1', 'block14_sepconv1']

    # NASNetLarge
    # end_layers_list = ['activation_238', 'activation_250', 'activation_259']

    # EfficientNetB5
    # end_layers_list = ['block6a_expand_activation', 'block7a_activation', 'top_activation']

    end_layers = [base.get_layer(i).output for i in end_layers_list]
    x = Conv2D(512, (1, 1), activation='relu')(end_layers[-1])  # last element of end_layers
    x = Dropout(0.8)(x)
    x = Conv2D(512, (1, 1), activation='relu')(x)
    x = Dropout(0.8)(x)

    # VGG16/19
    y = end_layers[0]
    z = end_layers[1]

    # EfficientNetB5
    # y = UpSampling2D(2)(end_layers[0])
    # y = Conv2D(1024, 1)(y)
    # z = UpSampling2D(2)(end_layers[1])
    # z = Conv2D(512, 1)(z)

    # Xception
    # y = UpSampling2D(2)(end_layers[0])
    # y = Conv2D(256, 1)(y)
    # z = end_layers[1]
    # z = Conv2D(512, 1)(z)

    # NASNetLarge
    # y = UpSampling2D(4)(end_layers[0])
    # y = Conv2D(1024, 1)(y)
    # z = UpSampling2D(2)(end_layers[1])
    # z = Conv2D(512, 1)(z)

    # DenseNet201
    # y = UpSampling2D(2)(end_layers[0])
    # z = UpSampling2D(2)(end_layers[1])

    # ResNet50
    # y = Conv2D(1024, 1)(end_layers[0])
    # z = Conv2D(512, 1)(end_layers[1])

    table_branch = TableConvLayer(name='table')([x, y, z])
    column_branch = ColumnConvLayer(name='columns')([x, y, z])

    mdl = Model(inputs=base.input, outputs=[table_branch, column_branch],
                name='ModelVGG19')
    return mdl


# funktion, welches die vorhersagen des netzes von shape (1, x, y) auf (x, y) umrechnet
def create_masks(msk):
    tf_msk = tf.argmax(msk, axis=3)
    tf_msk = tf_msk[..., tf.newaxis]
    return tf_msk[0]


# starten des skripts
if __name__ == "__main__":
    main()
